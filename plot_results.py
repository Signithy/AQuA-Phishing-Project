import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"


def normalize_variant(v):
    v = str(v).lower()
    if "raw" in v:
        return "raw"
    if "label" in v or "cl" in v or "clean" in v:
        return "labelclean"
    return v


def load_data(path):
    df = pd.read_csv(path)
    required = ["model", "train_dataset", "test_dataset", "train_variant", "accuracy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df["train_variant"] = df["train_variant"].apply(normalize_variant)
    return df


def plot_within_accuracy(df, output_path):
    within = df[df["train_dataset"] == df["test_dataset"]].copy()
    datasets = sorted(within["train_dataset"].unique())
    models = ["logreg", "bert"]
    variants = ["raw", "labelclean"]
    x = np.arange(len(datasets))
    width = 0.18
    fig, ax = plt.subplots()
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    labels = []
    for i, model in enumerate(models):
        for j, variant in enumerate(variants):
            mask = (within["model"] == model) & (within["train_variant"] == variant)
            group = within[mask]
            accs = []
            for d in datasets:
                row = group[group["train_dataset"] == d]
                if len(row) == 0:
                    accs.append(np.nan)
                else:
                    accs.append(row["accuracy"].iloc[0])
            pos = x + offsets[i * len(variants) + j]
            ax.bar(pos, accs, width)
            labels.append(f"{model}_{variant}")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy")
    ax.set_title("Within-dataset accuracy (LR vs BERT, raw vs labelclean)")
    ax.legend(labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def pivot_for_heatmap(df, model, variant):
    sub = df[(df["model"] == model) & (df["train_variant"] == variant)].copy()
    if sub.empty:
        return None
    pivot = sub.pivot_table(
        index="train_dataset",
        columns="test_dataset",
        values="accuracy",
        aggfunc="mean",
    )
    return pivot


def plot_heatmap(pivot, title, output_path):
    if pivot is None or pivot.empty:
        return
    fig, ax = plt.subplots()
    data = pivot.values
    ax.imshow(data)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = ""
            else:
                text = f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cross_heatmaps(df, outdir):
    cross = df[df["train_dataset"] != df["test_dataset"]].copy()
    models = ["logreg", "bert"]
    variants = ["raw", "labelclean"]
    for model in models:
        for variant in variants:
            pivot = pivot_for_heatmap(cross, model, variant)
            if pivot is None:
                continue
            title = f"Cross-dataset accuracy ({model}, {variant})"
            fname = f"cross_accuracy_{model}_{variant}.png"
            plot_heatmap(pivot, title, Path(outdir) / fname)


def plot_delta_accuracy(df, output_path):
    within = df[df["train_dataset"] == df["test_dataset"]].copy()
    models = ["logreg", "bert"]
    datasets = sorted(within["train_dataset"].unique())
    rows = []
    for model in models:
        for d in datasets:
            raw_rows = within[
                (within["model"] == model)
                & (within["train_dataset"] == d)
                & (within["train_variant"] == "raw")
            ]
            clean_rows = within[
                (within["model"] == model)
                & (within["train_dataset"] == d)
                & (within["train_variant"] == "labelclean")
            ]
            if len(raw_rows) == 0 or len(clean_rows) == 0:
                continue
            raw_acc = raw_rows["accuracy"].iloc[0]
            clean_acc = clean_rows["accuracy"].iloc[0]
            delta = clean_acc - raw_acc
            rows.append((model, d, delta))
    if not rows:
        return
    labels = [f"{m}_{d}" for m, d, _ in rows]
    deltas = [r[2] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, deltas)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Δ Accuracy (labelclean - raw)")
    ax.set_title("Change in within-dataset accuracy after CL cleaning")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="combined_results_table.csv",
        help="Combined results CSV (looked for in data/ by default)",
    )
    parser.add_argument(
        "--outdir",
        default=str(FIG_DIR),
        help="Directory to save figures (default figures/)",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / args.input_csv
    df = load_data(csv_path)
    plot_within_accuracy(df, outdir / "within_accuracy.png")
    plot_cross_heatmaps(df, outdir)
    plot_delta_accuracy(df, outdir / "delta_within_accuracy.png")


if __name__ == "__main__":
    main()
