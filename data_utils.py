import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from scipy import stats

def generate_synthetic_data(num_samples=1000, anomaly_ratio=0.1):
    categories = ['Groceries', 'Shopping', 'Bills', 'Transfer', 'Salary', 'Rent', 'Entertainment', 'Dining']
    
    data = []
    for _ in range(num_samples):
        is_anomaly = random.random() < anomaly_ratio
        category = random.choice(categories)
        amount = np.random.lognormal(4, 1.5) if not is_anomaly else np.random.lognormal(7, 2)
        
        if is_anomaly:
            description = f"Suspicious transaction {random.choice(['withdrawal', 'transfer', 'payment'])} {random.randint(1000, 9999)}"
        else:
            store = random.choice(['Store', 'Market', 'Shop', 'Mall', 'Center'])
            number = random.randint(1, 100)
            description = f"{category} at {store} {number}"
        
        data.append({
            'date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
            'amount': round(amount, 2),
            'category': category,
            'description': description,
            'is_anomaly': int(is_anomaly)
        })
    
    return pd.DataFrame(data)

def save_to_sqlite(df, db_path='transactions.db'):
    conn = sqlite3.connect(db_path)
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    conn.close()

def load_from_sqlite(db_path='transactions.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM transactions', conn)
    conn.close()
    return df

def detect_anomalies_zscore(df, column='amount', threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))
    return z_scores > threshold

def prepare_data(df, test_size=0.2, random_state=42):
    train_df = df.sample(frac=1-test_size, random_state=random_state)
    test_df = df.drop(train_df.index)
    
    train_texts = train_df['description'].tolist()
    test_texts = test_df['description'].tolist()
    
    label_map = {cat: i for i, cat in enumerate(df['category'].unique())}
    train_labels = [label_map[cat] for cat in train_df['category']]
    test_labels = [label_map[cat] for cat in test_df['category']]
    
    return {
        'train_texts': train_texts,
        'test_texts': test_texts,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'label_map': label_map
    }
