import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def resolve_data_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(DATA_DIR / path)


def run_logreg_experiment(train_csv, test_csv, text_col="text", label_col="label"):
    train_csv = resolve_data_path(train_csv)
    test_csv = resolve_data_path(test_csv)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    if text_col not in df_train.columns or text_col not in df_test.columns:
        raise ValueError(f"Text column '{text_col}' not found in both train and test CSVs.")
    if label_col not in df_train.columns or label_col not in df_test.columns:
        raise ValueError(f"Label column '{label_col}' not found in both train and test CSVs.")

    X_train_text = df_train[text_col].fillna("").astype(str)
    X_test_text = df_test[text_col].fillna("").astype(str)
    y_train = df_train[label_col].to_numpy()
    y_test = df_test[label_col].to_numpy()

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    args = parser.parse_args()

    metrics = run_logreg_experiment(
        args.train_csv,
        args.test_csv,
        text_col=args.text_col,
        label_col=args.label_col
    )

    print("Logistic Regression results")
    print(f"Train CSV: {args.train_csv}")
    print(f"Test CSV : {args.test_csv}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
