import torch
import torch.nn as nn
import torch.nn.functional as F


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
