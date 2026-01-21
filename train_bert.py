import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def resolve_data_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(DATA_DIR / path)


def load_datasets(train_csv, test_csv, text_col="text", label_col="label"):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    if text_col not in df_train.columns or text_col not in df_test.columns:
        raise ValueError(f"Text column '{text_col}' not found in both train and test.")
    if label_col not in df_train.columns or label_col not in df_test.columns:
        raise ValueError(f"Label column '{label_col}' not found in both train and test.")

    df_train = df_train[[text_col, label_col]].dropna()
    df_test = df_test[[text_col, label_col]].dropna()

    df_train = df_train.rename(columns={text_col: "text", label_col: "label"})
    df_test = df_test.rename(columns={text_col: "text", label_col: "label"})

    train_ds = Dataset.from_pandas(df_train, preserve_index=False)
    test_ds = Dataset.from_pandas(df_test, preserve_index=False)
    return train_ds, test_ds


def tokenize_datasets(train_ds, test_ds, tokenizer, max_length):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_tok = train_ds.map(tokenize_batch, batched=True)
    test_tok = test_ds.map(tokenize_batch, batched=True)

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return train_tok, test_tok


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_bert_experiment(
    train_csv,
    test_csv,
    text_col="text",
    label_col="label",
    model_name="distilbert-base-uncased",
    epochs=1,
    batch_size=8,
    learning_rate=5e-5,
    max_length=64,
):
    train_csv = resolve_data_path(train_csv)
    test_csv = resolve_data_path(test_csv)

    train_ds, test_ds = load_datasets(train_csv, test_csv, text_col=text_col, label_col=label_col)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_tok, test_tok = tokenize_datasets(train_ds, test_ds, tokenizer, max_length)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to("cpu")

    print("Using device: cpu")
    print(f"Train samples: {len(train_tok)}, Test samples: {len(test_tok)}")

    out_dir = os.path.join(
        BASE_DIR,
        "bert_runs",
        f"{os.path.splitext(os.path.basename(train_csv))[0]}_to_{os.path.splitext(os.path.basename(test_csv))[0]}",
    )
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        do_train=True,
        do_eval=False,
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    preds = trainer.predict(test_tok)
    metrics = compute_metrics((preds.predictions, preds.label_ids))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()

    metrics = run_bert_experiment(
        args.train_csv,
        args.test_csv,
        text_col=args.text_col,
        label_col=args.label_col,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("BERT results")
    print(f"Train CSV: {args.train_csv}")
    print(f"Test CSV : {args.test_csv}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
