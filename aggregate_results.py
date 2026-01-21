import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def normalize_variant(v):
    v = str(v).lower()
    if "raw" in v:
        return "raw"
    if "cl" in v or "label" in v or "clean" in v:
        return "labelclean"
    return "other"


def load_and_standardize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    name = path.name.lower()
    if "model" not in df.columns:
        if "bert" in name:
            df["model"] = "bert"
        elif "logreg" in name or "lr" in name:
            df["model"] = "logreg"
        else:
            df["model"] = "unknown"
    model = df["model"].iloc[0] if len(df) > 0 else "unknown"

    if model == "logreg":
        if "train_variant" in df.columns:
            tv = df["train_variant"]
        elif "train_version" in df.columns:
            tv = df["train_version"]
        else:
            tv = "raw"
        if "test_variant" in df.columns:
            te = df["test_variant"]
        elif "test_version" in df.columns:
            te = df["test_version"]
        else:
            te = "raw"
        df["train_variant"] = pd.Series(tv).apply(normalize_variant)
        df["test_variant"] = pd.Series(te).apply(normalize_variant)
    else:
        if "train_variant" in df.columns:
            tv = df["train_variant"]
        elif "train_label_quality" in df.columns:
            tv = df["train_label_quality"]
        else:
            tv = "raw"
        df["train_variant"] = pd.Series(tv).apply(normalize_variant)
        df["test_variant"] = "raw"

    return df


def resolve_data_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return DATA_DIR / path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LR and BERT results, keep only raw test and raw/CL train."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["logreg_results_table.csv", "bert_results_table.csv"],
        help="Input CSV files to merge (looked for in data/ by default).",
    )
    parser.add_argument(
        "--output",
        default="combined_results_table.csv",
        help="Output CSV file name (saved in data/ by default).",
    )
    args = parser.parse_args()

    dfs = []
    for p in args.inputs:
        path = resolve_data_path(p)
        if not path.is_file():
            print(f"Warning: {path} not found, skipping.")
            continue
        df = load_and_standardize(path)
        dfs.append(df)

    if not dfs:
        print("No valid input files found. Nothing to do.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    combined = combined[
        (combined["test_variant"] == "raw")
        & (combined["train_variant"].isin(["raw", "labelclean"]))
    ]

    out_path = resolve_data_path(args.output)
    combined.to_csv(out_path, index=False)
    print(f"Saved filtered combined results to {out_path}")
    print(f"Total rows: {len(combined)}")
    if "model" in combined.columns:
        print(combined["model"].value_counts())


if __name__ == "__main__":
    main()
