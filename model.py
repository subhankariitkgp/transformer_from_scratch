import torch 
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    


class PositioanlEmbeddings(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(self.max_seq_len, self.d_model) ##(max_seq_len, d_model)
        pos = torch.arange(0, self.max_seq_len).unsqueeze(1) ## (max_seq_len, 1)
        div_term = math.pow(10000, torch.arange(0, self.d_model, 2)/self.d_model)
        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)
        pe = pe.unsqueeze(0) ## (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x +  self.pe[:, :x.size(1), :].requires_grad(False)
        return self.dropout(x)



class LayerNormalization(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.var(-1, keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.beta