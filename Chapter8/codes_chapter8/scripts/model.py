import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF # Make sure to install torchcrf using 'pip install pytorch-crf'
from torch.utils.data import DataLoader, TensorDataset
import pickle

class BiLSTM_CRF(nn.Module):
    def __init__(self, word2idx, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx['PAD'])
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=5, bidirectional=True, batch_first=True, dropout=0.1)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.bilstm(embeddings)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask)

    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)
