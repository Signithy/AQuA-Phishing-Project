import os
import argparse
from pathlib import Path

import pandas as pd

from train_logreg import run_logreg_experiment


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
        default="logreg_results_table.csv",
        help="Output CSV for logistic regression results (saved in data/ by default)",
    )
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
    for train_key, train_file in datasets.items():
        train_full = resolve_data_path(train_file)
        if not os.path.isfile(train_full):
            print(f"Warning: train file {train_full} not found, skipping.")
            continue
        if train_key.endswith("_raw"):
            train_dataset = train_key.replace("_raw", "")
            train_variant = "raw"
        else:
            train_dataset = train_key.replace("_cl", "")
            train_variant = "labelclean"

        for test_key, test_file in datasets.items():
            if not test_key.endswith("_raw"):
                continue
            test_full = resolve_data_path(test_file)
            if not os.path.isfile(test_full):
                print(f"Warning: test file {test_full} not found, skipping.")
                continue
            test_dataset = test_key.replace("_raw", "")
            test_variant = "raw"

            print(
                f"Running LR: train={train_dataset} ({train_variant}), "
                f"test={test_dataset} ({test_variant})"
            )

            metrics = run_logreg_experiment(train_full, test_full)

            row = {
                "model": "logreg",
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "train_variant": train_variant,
                "test_variant": test_variant,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
            rows.append(row)

    if not rows:
        print("No runs completed; no rows to save.")
        return

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / args.output
    df.to_csv(out_path, index=False)
    print(f"Saved LR results to {out_path}")


if __name__ == "__main__":
    main()
