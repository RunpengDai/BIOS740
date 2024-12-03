import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF # Make sure to install torchcrf using 'pip install pytorch-crf'
from torch.utils.data import DataLoader, TensorDataset
import pickle
from model import BiLSTM_CRF

MAX_LEN = 128
EMBEDDING_DIM = 64
HIDDEN_DIM = 64

# Load datasets 
with open('../data/X_train.pt', 'rb') as f:
    X_train = torch.load(f)
with open('../data/y_train.pt', 'rb') as f:
    y_train = torch.load(f)
with open('../data/X_test.pt', 'rb') as f:
    X_test = torch.load(f)
with open('../data/y_test.pt', 'rb') as f:
    y_test = torch.load(f)

# Load mapping
with open('../data/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('../data/tag2idx.pkl', 'rb') as f:
    tag2idx = pickle.load(f)


# Create model instance
vocab_size = len(word2idx)
tagset_size = len(tag2idx)
model = BiLSTM_CRF(word2idx, vocab_size, tagset_size, EMBEDDING_DIM, HIDDEN_DIM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in train_loader:
        sentences, labels = batch
        sentences, labels = sentences.to(device), labels.to(device)
        mask = (sentences != word2idx['PAD']).to(device)
        optimizer.zero_grad()
        emissions = model(sentences)
        loss = model.loss(emissions, labels, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Save model
torch.save(model.state_dict(), '../data/bilstm_crf_model.pt')