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
              output_dir='./results', num_epochs=3, batch_size=16, learning_rate=2e-5):
        
        train_dataset = TransactionDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = TransactionDataset(test_texts, test_labels, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                

                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.device = device
        
    def predict(self, texts, batch_size=16):
        self.model.eval()
        dataset = TransactionDataset(texts, [0]*len(texts), self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, texts, labels, batch_size=16):
        predictions = self.predict(texts, batch_size)
        return {
            'f1': f1_score(labels, predictions, average='weighted'),
            'predictions': predictions
        }
