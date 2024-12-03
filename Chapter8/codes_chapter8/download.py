import os
import requests
import zipfile

# Create a directory to store the dataset
os.makedirs('data', exist_ok=True)

# URLs of the training and test datasets
train_url = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItrainset_corpus.zip'
test_url = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBItestset_corpus.zip'
dev_url = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBIdevelopset_corpus.zip'

# Download the training dataset
train_response = requests.get(train_url)
with open('data/NCBI_training_data.zip', 'wb') as f:
    f.write(train_response.content)

# Download the test dataset
test_response = requests.get(test_url)
with open('data/NCBI_test_data.zip', 'wb') as f:
    f.write(test_response.content)

# Download the development dataset
dev_response = requests.get(test_url)
with open('data/NCBI_dev_data.zip', 'wb') as f:
    f.write(dev_response.content)

# Unzip the datasets
with zipfile.ZipFile('data/NCBI_training_data.zip', 'r') as zip_ref:
    zip_ref.extractall('data/NCBI_corpus_training.txt')

with zipfile.ZipFile('data/NCBI_test_data.zip', 'r') as zip_ref:
    zip_ref.extractall('data/NCBI_corpus_testing.txt')

with zipfile.ZipFile('data/NCBI_dev_data.zip', 'r') as zip_ref:
    zip_ref.extractall('data/NCBI_corpus_development.txt')
