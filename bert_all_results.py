import argparse
from pathlib import Path

import pandas as pd

from train_bert import run_bert_experiment


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def resolve_data_path(p: str) -> str:
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str(DATA_DIR / path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="bert_results_table.csv",
        help="Output CSV for BERT results (saved in data/ by default).",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    datasets = {
        "enron_raw": "enron_raw.csv",
        "naser_raw": "naser_raw.csv",
        "twente_raw": "twente_raw.csv",
        "enron_cl": "enron_cl_labelclean.csv",
        "naser_cl": "naser_cl_labelclean.csv",
        "twente_cl": "twente_cl_labelclean.csv",
    }

    rows = []

    for train_key, train_csv in datasets.items():
        for test_key, test_csv in datasets.items():
            if not test_key.endswith("_raw"):
                continue

            train_dataset = train_key.split("_")[0]
            test_dataset = test_key.split("_")[0]
            train_quality = "cl_cleaned" if train_key.endswith("_cl") else "raw"

            print("===== TRAINING BERT =====")
            print(f"Train: {train_csv} ({train_quality})")
            print(f"Test : {test_csv} (raw)")

            metrics = run_bert_experiment(
                train_csv=resolve_data_path(train_csv),
                test_csv=resolve_data_path(test_csv),
                text_col="text",
                label_col="label",
                model_name="distilbert-base-uncased",
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )

            row = {
                "model": "bert",
                "train_key": train_key,
                "test_key": test_key,
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "train_label_quality": train_quality,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / args.output
    df.to_csv(out_path, index=False)
    print(f"Saved BERT results to {out_path}")


if __name__ == "__main__":
    main()
