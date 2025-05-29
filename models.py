import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

class TransactionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class ZeroShotClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.clf = LogisticRegression(max_iter=1000)
        
    def train(self, train_texts, train_labels):
        train_embeddings = self.model.encode(train_texts)
        self.clf.fit(train_embeddings, train_labels)
        
    def predict(self, texts):
        embeddings = self.model.encode(texts)
        return self.clf.predict(embeddings)
    
    def evaluate(self, texts, labels):
        preds = self.predict(texts)
        return {
            'f1': f1_score(labels, preds, average='weighted'),
            'predictions': preds
        }

class FineTunedClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=8):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        
    def train(self, train_texts, train_labels, test_texts, test_labels, 
              output_dir='./results', num_epochs=3, batch_size=16):
        
        train_dataset = TransactionDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = TransactionDataset(test_texts, test_labels, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            max_grad_norm=1.0,
            gradient_accumulation_steps=2,
            fp16=torch.cuda.is_available(),
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        trainer.train()
        self.trainer = trainer
        
    def predict(self, texts):
        dataset = TransactionDataset(texts, [0]*len(texts), self.tokenizer)
        preds = self.trainer.predict(dataset)
        return np.argmax(preds.predictions, axis=1)
    
    def evaluate(self, texts, labels):
        preds = self.predict(texts)
        return {
            'f1': f1_score(labels, preds, average='weighted'),
            'predictions': preds
        }
