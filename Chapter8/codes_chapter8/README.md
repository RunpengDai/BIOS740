# Disease Named Entity Recognition with Bi-LSTM-CRF

This project implements a Named Entity Recognition (NER) system to identify disease mentions in biomedical texts using the NCBI Disease Corpus.

## Requirements

pip install torch pytorch-crf tqdm spacy pickle eqeval

## Setup

### **Install Dependencies**:

```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
```

### **Download the Dataset**:

Place NCBI_corpus_training.txt, NCBI_corpus_development.txt, and NCBI_corpus_testing.txt into the data/ directory.
Data Preprocessing:

    ``bash     python scripts/preprocess_data.py     ``

### Train the Model: (Note that there is large room for improvement, model, epochs, ... etc)

    ``bash     python scripts/train_model.py     ``

### **Evaluate the Model**:

    ``bash     python scripts/evaluate_model.py     ``

### **Results**

F1-Score: To be updated after evaluation.
Precision: To be updated after evaluation.
Recall: To be updated after evaluation.
