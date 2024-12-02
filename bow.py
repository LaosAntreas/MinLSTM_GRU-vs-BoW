import torch
import torch.nn as nn

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
 
