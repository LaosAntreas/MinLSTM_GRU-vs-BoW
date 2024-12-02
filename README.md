# MinLSTM_GRU-vs-BoW
Comparing MinLSTM/MinGRU with Bag of Words on DAIR.AI's emotions dataset

---


## The task at hand

The objective of this project is: Given a text sequence(tweet) from DAIR-AI's emotion dataset, determine wether said text falls under one of the following classes: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).  
We decided to implement this using two models:
- MinLSTM/MinGRU as proposed in the paper [Were RNNs All We Needed?](https://arxiv.org/pdf/2410.01201)
- and a simple Bag of Words model 

The problem can be broken down into the following tasks:

### 1. Data Visualisation

Before creating the model, we first need to understand our data to try to see if we have any imbalances in the distribution of the classes. Here we will also visualise the class distributions of train, test and validation datasets individually to check if there are any deviations between them. 

<img width="351" alt="Screenshot 2024-12-02 at 4 18 46 PM" src="https://github.com/user-attachments/assets/6140169b-5c8a-4888-8cea-6fc590912be6">
<img width="350" alt="Screenshot 2024-12-02 at 4 19 09 PM" src="https://github.com/user-attachments/assets/dac09109-db32-4291-8b53-7bd52e82f880">
<img width="349" alt="Screenshot 2024-12-02 at 4 19 23 PM" src="https://github.com/user-attachments/assets/6ab62796-1881-421c-b23d-deb804b5f22f">

It seems that joy and sadness are overrepresented in our data but training, test and validation seem to have the same class distributions (deviations between them can be considered statistically insignificant).

Here, we also visualised the sequence length for each of the individual data samples. There will be no effect on the models we chose since they are both context length independent but it is still nice to visualise.

<img width="474" alt="Screenshot 2024-12-02 at 4 20 02 PM" src="https://github.com/user-attachments/assets/3d0ed39c-c6c1-4407-97be-431500dbaf00">

- Range: 2 - 66 
- Mean: 19.14 
- Standard Deviation: 10.97

Here it is interesting to note, that the plot follows a Gamma distribution.

As a baseline, it's good to see how a model with random predictions would perform:

Intuitively, since 6 classes exist it would suggest that the model's accuracy is $1/6 = 0.1\dot{6}$, but since the distribution of classes isn't uniform, we also ran a Monte Carlo simulation as a sanity check:
 
```python
#monte carlo
correct = 0
total = 0
for i in range(10000):
	pred = np.random.randint(0, 6, len(test_labels))
	correct += np.sum(pred == test_labels)
	total += len(test_labels)
accuracy = correct/total*100
print(f'Accuracy: {accuracy:.2f}%')
```

The output was: `Accuracy: 16.66%`  which matches our hypothesis.

Another good comparison is with a model that only predicts the most common seen class. In that case the accuracy would be 34.8% based on the previous plots.

---
## 2. Data Loading and Pre-Processing

Since the data straight from the library is in the format: dictionary of lists, we must first convert it into a usable format for `pytorch`. Here we decided do use a custom `torch.Dataset` for the conversion. To do that though we first need to create a tokeniser to convert the text into a format that the model would understand. For this, we decided to use the `tokenizer` library and tokenise by whitespace as there is no need for more complex tokeniser methods for such datasets and models. 

```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def create_tokenizer(text, vocab_size):
    tokenizer = Tokenizer(WordLevel(unk_token="<OOV>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=["<OOV>", "<PAD>"], min_frequency=1)
    tokenizer.train_from_iterator(text, trainer)
    return tokenizer
```

We also added two additional tokens, "\<OOV>" for out of vocabulary tokens and "\<PAD>" to pad the sequences to have the same size as the rest in the batch, we do this because `pytorch` does not handle dynamic sequence lengths for sequences in the same batch.

Having this, we can move on to creating the dataset.

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    A self-defined collate function for padding the sequences before feeding them into the model
    """
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    return padded_sequences, torch.tensor(labels, dtype=torch.long)

class EmotionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences  
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        sentence = self.tokenizer.encode(sentence).ids
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long)
```

The collate function ensures that we pad the sequences of a batch so that they have the same length and the dataset class uses the tokeniser to convert text into a sequence of one hot vectors. 

```python
tokenizer = create_tokenizer(ds['train']['text'], len(ds['train']['text']))

train = EmotionDataset(ds['train']['text'], ds['train']['label'], tokenizer)
test = EmotionDataset(ds['test']['text'], ds['test']['label'], tokenizer)
val = EmotionDataset(ds['validation']['text'], ds['validation']['label'], tokenizer)

testLoader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True, collate_fn=collate_fn)
trainLoader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate_fn)
valLoader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=True, collate_fn=collate_fn)
```

Some things to note here:
- Batch size is set to 64 because that was the size set in the original paper
- Vocabulary size is set to the length of the training dataset. This was done as a way to scale it to the size of any other dataset in our case `vocab_size = 16000`

---
## 3. Creating the models

#### RNN models

Here we decided to implement the recent MinLSTM and MinGRU models as they exhibit some nice properties. Firstly, in contrast with traditional RNNs these models are parallelizable this means they can be efficiently trained with the use of a GPU. Additionally, these models scale linearly with the size of the input making them more efficient than transformer based models which require $O(n^2)$ complexity. 

**MinGRU**

```python

#Helper functions
def parallel_scan_log(log_coeffs, log_values):
    # log_coeffs: (batch_size, seq_len, units)
    # log_values: (batch_size, seq_len + 1, units)
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp( log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]

def log_g(x):
    return torch.where(x >= 0, (F.relu(x)+0.5).log(), -F.softplus(-x))

class MinGRUCell_log(nn.Module):
    '''Implements minGRU cell in log-space'''
    def __init__(self, hidden_size, embedding_size):
        super(MinGRUCell_log, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.linear_z = nn.Linear(self.embedding_size, self.hidden_size)
        self.linear_h = nn.Linear(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, x, h_0):
        '''
        Forward pass of the minGRU cell in log-space
        Args:
            x: input sequence          (batch_size, seq_len, embedding_size)
            h_0: initial hidden state  (batch_size, 1, hidden_size)

        Returns:
            out: (batch_size, seq_len, hidden_size) model output
            h: (batch_size, 1, hidden_size) hidden state

        '''
        k = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        
        h = parallel_scan_log(log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1))
        return self.out(h), h[:,-1,:].unsqueeze(1) # (batch_size, seq_len, embedding_size)

```

**MinLSTM**

```python
class MinLSTMCell_log(nn.Module):
    '''Implements minLSTM cell in log-space'''
    def __init__(self, hidden_size, embedding_size):    
        super(MinLSTMCell_log, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.linear_f = nn.Linear(self.embedding_size, self.hidden_size)
        self.linear_i = nn.Linear(self.embedding_size, self.hidden_size)
        self.linear_h = nn.Linear(self.embedding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.embedding_size)

    def forward(self, x, h_0):
        '''
        Forward pass of the minLSTM cell in log-space
        Args:
            x: input sequence          (batch_size, seq_len, ebmedding_size)
            h_0: initial hidden state  (batch_size, 1, hidden_size)

        Returns:
            out: (batch_size, seq_len, embedding_size) model output
            h: (batch_size, 1, embedding_size) hidden state
        '''
        diff = F.softplus(-self.linear_f(x))  - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        return self.out(h), h[:, -1, :].unsqueeze(1) 

```

We designed both cells proposed in the paper like above. The models parallelism is based on the `parallel_scan_log()` function which is in turn based on an algorithm proposed in the paper [Efficient Parallelization of a Ubiquitous Sequential Computation](https://arxiv.org/pdf/2311.06281). 

Based again on the paper, we designed a layer block used specifically for language modelling. 

```python
class minRNNLayer(nn.Module):
    '''
    Implements a single layer of minRNN as described in the paper
    '''
    def __init__(self, hidden_size, embedding_size, cell_type = 'lstm', dropout=0.2):
        super(minRNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.cell_type = cell_type

        if cell_type == 'lstm':
            self.cell = MinLSTMCell_log(hidden_size, embedding_size)
        elif cell_type == 'gru':
            self.cell = MinGRUCell_log(hidden_size, embedding_size)
        else:
            raise ValueError('Invalid cell type. Choose between "lstm" and "gru"')
        
        self.conv1 = nn.Conv1d(embedding_size, embedding_size, 4, padding='same') 
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)  

        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size*4),
            self.dropout,
            nn.ReLU(),
            nn.Linear(embedding_size*4, embedding_size)
        )
    
    def forward(self, x, h_0):
        """
        Args:
            x: (batch_size, sentence_length, embedding_size)
            h_0: (batch_size, 1, hidden_size)

        output:
            x: (batch_size, sentence_length, hidden_size) layer output
            h: (batch_size, 1, hidden_size) hidden state

        """
        x = self.layer_norm(x) # (batch_size, sentence_length, embedding_size)
        x = self.dropout(x)
        out, h = self.cell(x, h_0) # (batch_size, sentence_length, embedding_size)
        x = x + out

        x = self.layer_norm(x) # (batch_size, sentence_length, embedding_size)
        x = self.dropout(x)

        x = x + self.conv1(x.permute(0,2,1)).permute(0,2,1) # (batch_size, sentence_length, embedding_size)
        x = self.layer_norm(x) # (batch_size, sentence_length, embedding_size)
        x = self.dropout(x)
        
        x = x + self.mlp(x) # (batch_size, sentence_length, embedding_size)
        #print("layer output",h.shape)   
        return x, h
```

The layer consists of the RNN cell, a convolutional layer and a 2 layer MLP. We use residual connections (to help with stability) and dropout (for overfitting) between each layer. 

Finally, the complete model is created as follows:

```python
class minRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, output_size, cell_type="lstm", dropout=0.2, num_layers=3):
        super(minRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.output_size = output_size



        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn_layers = nn.ModuleList([minRNNLayer(hidden_size, embedding_size, cell_type= cell_type, dropout=dropout) for _ in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        '''
        Args:
            x: (batch_size, sentence_length)
        
        Returns:
            (batch_size, output_size)
        '''
        x = self.embedding(x) # (batch_size, sentence_length, embedding_size)
        h_0 = torch.zeros(x.size(0), 1, self.hidden_size).to(x.device)
        for layer in self.rnn_layers:
            x_t, h_0 = layer(x, h_0)
            x = x + x_t
        x = self.classifier(x[:,-1,:]) # (batch_size, output_size)
        return x
```

Many layers can be stacked on top of each other. Taking in the tokenized input, we convert in into a more manageable size with the use of `nn.Embedding`. We then go through the layers, inputting an initial hidden state of zeroes and then use a classifier in the end. Again, we use residual connections for the layers to ensure computational stability.

#### Bag of Words Model

Model number two is a *lot* simpler than model one. We decided to do this as we believe that BoW is the simplest non trivial example of a model that can be used for such a task. This, will give us a good comparison on how beneficial is the additional complexity of MinRNN is. 

```python
class BagOfWords(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.25):
        super(BagOfWords, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, sentence_length)
        
        Returns:
            (batch_size, output_dim)
        """
        x = self.embedding(x) # (batch_size, sentence_length, embedding_dim)
        x = torch.mean(x, dim=1) # (batch_size, embedding_dim)
        x = self.mlp(x)
        return x
```

There is not much to be said about this model. Again we used `nn.Embedding` to reduce the size of the input to a more manageable size. After this, `torch.mean` is used over the sequence dimension to create a Bag-of-Words representation and then a simple MLP with dropout to improve performance.

---
## 4. Training the models

Both models were trained using an early stopping module to ensure no overfitting occurred. 

```python
class EarlyStopping():
    def __init__(self, patience=3, min_delta=0.0, path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.model_states = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.path is not None:
          torch.save(model.state_dict(), self.path)
        else: 
          self.model_states = model.state_dict()  
```

`path=None` was used when training on Colab. Since saving data to the drive is extremely slow and we have enough ram, rather than saving the model weights we preferred storing them in memory. 
#### Hyperparameters
- MinRNN
	- Embedding size: 256
	- Hidden size: 512 
	- Epochs: 50 (with early stopping)
	- Cell Type: LSTM, it was chosen as it exhibited better performance than MinGRU in the paper
	- Layers: 2 (Better than 1 and the same performance as 3)
	- Optimiser: AdamW 
	- Learning rate: 0.001
	- Early Stopping patience: 3
	- Evaluation every 2 Epochs
	- Dropout: 0.2 (as mentioned in the paper)
- BagOfWords
	- Embedding size: 256
	- Hidden size: 512 
	- Epochs: 30 (with early stopping)
	- Optimiser: AdamW
	- Learning rate: 0.001
	- Early Stopping patience: 3
	- Evaluations every 2 epochs
	- Dropout: 0.25

## 5. Evaluation

### MinLSTM
- ##### Training and Validation Losses
	- <img width="454" alt="Screenshot 2024-12-02 at 4 20 48 PM" src="https://github.com/user-attachments/assets/7c93316d-42a2-4b16-a9b2-f2128964ac0d">

		- The model has converged in the 10th epoch (since losses are logged at every second epoch)
- #### Test Accuracy: `91.2%`
- ##### Confusion Matrix
	- <img width="457" alt="Screenshot 2024-12-02 at 4 21 10 PM" src="https://github.com/user-attachments/assets/79ef14d2-bd2e-4430-8e12-f45c92244229">

	- Some notes on the confusion matrix:
		- The accuracy of each class seems to directly correlate with the fraction that the class takes in the training data. 
		- Positive emotions are more likely to be confused with other positive emotions (e.g., joy and love), while negative emotions tend to be confused with other negative emotions(e.g. fear and anger)
		- Surprise, which can be considered a neutral emotion and was also the least represented in the dataset is confused with both positive and negative emotions (fear and joy).
- #####  Sample Incorrect Predictions
```
Predicted: fear, True: sadness, Text: i feel quite helpless in all of this so prayer is the most effective tool i have because i have no answers and there is nothing else i can offer them right now 

Predicted: love, True: joy, Text: i feel more faithful than ever 

Predicted: joy, True: anger, Text: whenever i put myself in others shoes and try to make the person happy
```
- Intuitively, if the model sees a word or sequence of words that is mainly used in one class then it makes sense that it would classify the text as that class. In the last prediction, for example, the word "happy" is mainly associated with the joy class so it misclassifies the text.

### Bag of Words
- ##### Training and Validation Losses
	 - <img width="454" alt="Screenshot 2024-12-02 at 4 21 38 PM" src="https://github.com/user-attachments/assets/b54cf4bc-f87f-4ae4-8e80-f581b0fd1aaf">

		 - The model seems to have converged at the 8th epoch. A bit faster than MinLSTM.
- #### Test Accuracy: `85.5%`
- ##### Confusion Matrix
- <img width="456" alt="Screenshot 2024-12-02 at 4 22 11 PM" src="https://github.com/user-attachments/assets/5d756b63-f961-4127-8513-81e386f68bbe">

	- The hypothesis of better performance with greater representation in the dataset seems to hold with the Bag of Words model as well
	- Surprise is better predicted in this model rather than MinLSTM but the other classes have worse accuracy. 
	- The classes being confused seem to be the same as with MinLSTM.
- ##### Sample Incorrect Predictions
```
Predicted: surprise, True: joy, Text: i started walking again yesterday and it feels amazing

Predicted: sadness, True: anger, Text: i have been sitting at home today and all in all feeling quite stressed

Predicted: anger, True: sadness, Text: i woke up feeling groggy and grumpy and like the last thing i wanted to do was make dinner

Predicted: joy, True: anger, Text: i feel like i should be listening to and working on my mandarin but what i really want to listen to is the savage love or car talk
```
- As before, the model is more likely to classify a text belonging to a specific class if it sees a word that is often used in said class. In the first example, we can safely assume that 'amazing' is often used in the surprise class so, it misclassifies it as such.


