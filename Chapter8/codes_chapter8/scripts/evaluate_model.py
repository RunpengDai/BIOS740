from seqeval.metrics import classification_report, f1_score
import torch
from torchcrf import CRF
from model import BiLSTM_CRF
import pickle
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load mapping
with open('../data/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
    
with open('../data/tag2idx.pkl', 'rb') as f:
    tag2idx = pickle.load(f)

vocab_size = len(word2idx)
tagset_size = len(tag2idx)

EMBEDDING_DIM = 64
HIDDEN_DIM = 64

model = BiLSTM_CRF(word2idx, vocab_size, tagset_size, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('../data/bilstm_crf_model.pt'))
model.to(device)

# Load test data
with open('../data/X_test.pt', 'rb') as f:
    X_test = torch.load(f)


with open('../data/y_test.pt', 'rb') as f:
    y_test = torch.load(f)

# Predict tags
model.eval()
y_pred = model(X_test).detach().numpy()
y_pred = np.argmax(y_pred, axis=-1).tolist()
y_true = y_test.detach().tolist()

# Convert indices to tags
idx2tag = {v: k for k, v in tag2idx.items()}
y_pred_tags = [[idx2tag[idx] for idx in row] for row in y_pred]
y_true_tags = [[idx2tag[idx] for idx in row] for row in y_true]

# Remove padding
def remove_padding(sequences):
    cleaned_sequences = []
    for seq in sequences:
        cleaned_seq = []
        for tag in seq:
            if tag != 'O' or tag != 'PAD':
                cleaned_seq.append(tag)
        cleaned_sequences.append(cleaned_seq)
    return cleaned_sequences

y_pred_clean = remove_padding(y_pred_tags)
y_true_clean = remove_padding(y_true_tags)

# Print classification report
print(classification_report(y_true_clean, y_pred_clean))

# Sample error analysis
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if y_true_tags[i][j] != y_pred_tags[i][j]:
            word = list(word2idx.keys())[list(word2idx.values()).index(X_test[i][j])]
            print(f"Word: {word}, True Tag: {y_true_tags[i][j]}, Predicted Tag: {y_pred_tags[i][j]}")