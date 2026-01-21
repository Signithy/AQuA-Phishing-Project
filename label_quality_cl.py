import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_scored_csv", type=str, required=True)
    parser.add_argument("--output_clean_csv", type=str, required=True)
    parser.add_argument("--drop_fraction", type=float, default=0.1)
    parser.add_argument("--kfolds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # Expect at least these columns; extra columns (like loss, is_dropped) are fine.
    for col in ["id", "text", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    texts = df["text"].fillna("").astype(str)
    y = df["label"].to_numpy()

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2
    )
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )

    # Out-of-sample predicted probabilities for cleanlab
    pred_probs = cross_val_predict(
        clf,
        X,
        y,
        cv=args.kfolds,
        method="predict_proba",
        n_jobs=-1
    )

   
    issues = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence"
    )

    n = len(df)
    n_remove = max(1, int(args.drop_fraction * n))

    mask_noisy = np.zeros(n, dtype=bool)
    mask_noisy[issues[:n_remove]] = True
    mask_clean = ~mask_noisy

    df["cl_noisy"] = mask_noisy.astype(int)

    print(f"Total samples: {n}")
    print(f"Flagged as potential label errors (CL): {mask_noisy.sum()} ({mask_noisy.sum() / n:.3f})")
    print(f"Remaining clean samples: {mask_clean.sum()}")

    df.to_csv(args.output_scored_csv, index=False)
    df[mask_clean].to_csv(args.output_clean_csv, index=False)


if __name__ == "__main__":
    main()
