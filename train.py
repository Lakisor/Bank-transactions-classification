import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from data_utils import generate_synthetic_data, save_to_sqlite, load_from_sqlite, prepare_data, detect_anomalies_zscore
from models import ZeroShotClassifier, FineTunedClassifier, TransactionDataset
import config as cfg

def plot_results(true_labels, zero_shot_preds, fine_tuned_preds, label_map):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    idx_to_label = {v: k for k, v in label_map.items()}
    
    zero_shot_df = pd.crosstab(
        pd.Series([idx_to_label[l] for l in true_labels], name='True'),
        pd.Series([idx_to_label[p] for p in zero_shot_preds], name='Predicted')
    )
    sns.heatmap(zero_shot_df, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Zero-shot BERT')
    
    fine_tuned_df = pd.crosstab(
        pd.Series([idx_to_label[l] for l in true_labels], name='True'),
        pd.Series([idx_to_label[p] for p in fine_tuned_preds], name='Predicted')
    )
    sns.heatmap(fine_tuned_df, annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Fine-tuned BERT')
    
    plt.tight_layout()
    
    os.makedirs('output', exist_ok=True)
    plot_path = 'output/confusion_matrices.png'
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def main():
    mlflow.set_experiment('BankTransactionClassification')
    
    print("Generating synthetic data...")
    df = generate_synthetic_data(
        num_samples=cfg.NUM_SAMPLES,
        anomaly_ratio=cfg.ANOMALY_RATIO
    )
    
    print("Saving data to database...")
    save_to_sqlite(df, cfg.DATABASE_PATH)
    df = load_from_sqlite(cfg.DATABASE_PATH)
    
    df['is_anomaly_zscore'] = detect_anomalies_zscore(df)
    
    print("Preparing data for training...")
    data = prepare_data(
        df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE
    )
    
    with mlflow.start_run():
        print("Training Zero-shot model...")
        zero_shot = ZeroShotClassifier(model_name=cfg.ZERO_SHOT_MODEL)
        zero_shot.train(data['train_texts'], data['train_labels'])
        zero_shot_results = zero_shot.evaluate(data['test_texts'], data['test_labels'])
        
        print("Training Fine-tuned model...")
        fine_tuned = FineTunedClassifier(
            model_name=cfg.FINE_TUNED_MODEL,
            num_labels=len(data['label_map'])
        )
        fine_tuned.train(
            data['train_texts'], data['train_labels'],
            data['test_texts'], data['test_labels'],
            output_dir=cfg.OUTPUT_DIR,
            num_epochs=cfg.NUM_EPOCHS,
            batch_size=cfg.BATCH_SIZE
        )
        fine_tuned_results = fine_tuned.evaluate(data['test_texts'], data['test_labels'])
        
        mlflow.log_metric('zero_shot_f1', zero_shot_results['f1'])
        mlflow.log_metric('fine_tuned_f1', fine_tuned_results['f1'])
        
        plot_path = plot_results(
            data['test_labels'],
            zero_shot_results['predictions'],
            fine_tuned_results['predictions'],
            data['label_map']
        )
        mlflow.log_artifact(plot_path)
        
        print(f"\nResults:")
        print(f"Zero-shot F1: {zero_shot_results['f1']:.4f}")
        print(f"Fine-tuned F1: {fine_tuned_results['f1']:.4f}")

if __name__ == "__main__":
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    main()
