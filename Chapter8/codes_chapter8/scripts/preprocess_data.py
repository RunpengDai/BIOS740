import os
import spacy
import pickle
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

def parse_pubtator(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        articles = content.split('\n\n')
        for article in tqdm(articles):
            lines = article.strip().split('\n')
            text = ''
            annotations = []
            for line in lines:
                if '|t|' in line or '|a|' in line:
                    # Extract text
                    _, _, line_content = line.partition('|t|') if '|t|' in line else line.partition('|a|')
                    text += line_content + ' '
                elif line:
                    # Extract annotations
                    parts = line.split('\t')
                    if len(parts) == 6:
                        start = int(parts[1])
                        end = int(parts[2])
                        mention = parts[3]
                        entity = parts[4]
                        annotations.append({'start': start, 'end': end, 'mention': mention, 'entity': entity})
            if text:
                doc = nlp(text.strip())
                tokens = [token.text for token in doc]
                token_indices = [(token.idx, token.idx + len(token.text)) for token in doc]
                tags = ['O'] * len(tokens)
                for ann in annotations:
                    for idx, (start_idx, end_idx) in enumerate(token_indices):
                        if start_idx == ann['start']:
                            tags[idx] = 'B-' + ann['entity']
                        elif start_idx > ann['start'] and end_idx <= ann['end']:
                            tags[idx] = 'I-' + ann['entity']
                sentences.append((tokens, tags))
    return sentences

# Parse training and test data
train_sentences = parse_pubtator('../data/NCBItrainset_corpus.txt')
test_sentences = parse_pubtator('../data/NCBItestset_corpus.txt')

# Build word and tag vocabularies
words = set()
tags = set()
for tokens, label_tags in train_sentences:
    words.update(tokens)
    tags.update(label_tags)

word2idx = {word: idx + 2 for idx, word in enumerate(words)}
word2idx['PAD'] = 0
word2idx['UNK'] = 1

tag2idx = {tag: idx for idx, tag in enumerate(tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# Save mappings
with open('../data/word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)

with open('../data/tag2idx.pkl', 'wb') as f:
    pickle.dump(tag2idx, f)

# Converting Tokens and Tags to Indices:
MAX_LEN = 128


def prepare_sequences(sentences, word2idx, tag2idx):
    X = []
    y = []
    for tokens, tags in sentences:
        seq = [word2idx.get(token, word2idx['UNK']) for token in tokens]
        label_seq = [tag2idx[tag] for tag in tags]
        X.append(torch.tensor(seq, dtype=torch.long))
        y.append(torch.tensor(label_seq, dtype=torch.long))
    X = pad_sequence(X, batch_first=True, padding_value=word2idx['PAD'])
    y = pad_sequence(y, batch_first=True, padding_value=tag2idx['O'])
    # Truncate sequences to MAX_LEN
    X = X[:, :MAX_LEN]
    y = y[:, :MAX_LEN]
    return X, y


# Prepare training and test sequences
X_train, y_train = prepare_sequences(train_sentences, word2idx, tag2idx)
X_test, y_test = prepare_sequences(test_sentences, word2idx, tag2idx)

# Example of moving data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Save datasets
with open('../data/X_train.pt', 'wb') as f:
    torch.save(X_train, f)
with open('../data/y_train.pt', 'wb') as f:
    torch.save(y_train, f)
with open('../data/X_test.pt', 'wb') as f:
    torch.save(X_test, f)
with open('../data/y_test.pt', 'wb') as f:
    torch.save(y_test, f)
