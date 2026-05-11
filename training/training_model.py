import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

Model_Name = 'roberta-base'
Max_Length = 128  # max token sequence length
Train_Path = r'C:\Users\AmrAhmed\Documents\GitHub\Plagiarism-Guard\data\procressed\mrpc_train_clean.csv'
Test_Path = r'C:\Users\AmrAhmed\Documents\GitHub\Plagiarism-Guard\data\procressed\mrpc_test_clean.csv'
Save_Path = 'saved_models/saved_model'


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna().reset_index(drop=True)
    df['text1'] = df['text1'].astype('string').str.strip()
    df['text2'] = df['text2'].astype('string').str.strip()
    df['label'] = df['label'].astype('int')
    return df[['text1', 'text2', 'label']]

def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch['text1'],
        batch['text2'],
        truncation=True,
        padding='max_length',
        max_length=Max_Length
    )

def evaluate_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

def main():
    train_df = load_data(Train_Path)
    test_df = load_data(Test_Path)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(Model_Name)

    train_dataset = train_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True)

    train_dataset = train_dataset.remove_columns(['text1', 'text2'])
    test_dataset = test_dataset.remove_columns(['text1', 'text2'])

    train_dataset = train_dataset.rename_column('label', 'labels')
    test_dataset = test_dataset.rename_column('label', 'labels')

    train_dataset.set_format('torch')
    test_dataset.set_format('torch')

    model = AutoModelForSequenceClassification.from_pretrained(Model_Name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.02,
        logging_dir="logs",
        load_best_model_at_end=True,      # restore best checkpoint after training
        metric_for_best_model="f1_score", # optimise for F1
        greater_is_better=True,
        fp16=True,                         # mixed precision training
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=evaluate_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()
    metrics = trainer.evaluate()

    model.save_pretrained(Save_Path)
    tokenizer.save_pretrained(Save_Path)

    print('Saved model to: ', Save_Path)
    print('Final Metrics: ', metrics)

if __name__ == "__main__":
    main()