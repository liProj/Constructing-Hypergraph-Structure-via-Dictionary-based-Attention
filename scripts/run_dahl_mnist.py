from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from da_hypergraph.attention import DictionaryAttention
from da_hypergraph.data import balanced_paper_split, load_mnist
from da_hypergraph.hypergraph import build_hypergraph, hypergraph_label_propagation


ROOT = Path(__file__).resolve().parents[1]


def run(repeats: int, max_iter: int) -> list[dict[str, object]]:
    all_x, all_y = load_mnist(ROOT / "data" / "raw" / "mnist")
    rows: list[dict[str, object]] = []
    for seed in range(repeats):
        train_global, test_global = balanced_paper_split(all_y, random_state=seed)
        chosen = np.concatenate([train_global, test_global])
        x = all_x[chosen]
        y = all_y[chosen]
        train = np.arange(len(train_global))
        test = np.arange(len(train_global), len(chosen))

        for method in ("knn", "clustering", "representation"):
            h, centroids = build_hypergraph(
                x,
                method,
                k=10,
                n_clusters=10,
                random_state=seed,
            )
            baseline, _ = hypergraph_label_propagation(h, y, train, 10, regularization=0.3)
            h_attention = DictionaryAttention(
                n_components=50,
                alpha=2 ** -4,
                max_iter=max_iter,
                random_state=seed,
                stabilize=True,
            ).transform(x, h, centroids)
            attended, _ = hypergraph_label_propagation(h_attention, y, train, 10, regularization=0.3)
            rows.append(
                {
                    "seed": seed,
                    "generator": method,
                    "hl_accuracy": float((baseline[test] == y[test]).mean()),
                    "da_hl_accuracy": float((attended[test] == y[test]).mean()),
                    "delta": float((attended[test] == y[test]).mean() - (baseline[test] == y[test]).mean()),
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "dictionary_size": 50,
                    "alpha": 2 ** -4,
                    "lambda": 0.3,
                    "attention_policy": "cosine_clip_[eps,1]",
                }
            )
            print(rows[-1])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=12)
    args = parser.parse_args()
    rows = run(args.repeats, args.max_iter)

    output = ROOT / "results" / "reproduced" / "mnist_dahl.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nMean accuracies (%)")
    for method in ("knn", "clustering", "representation"):
        subset = [row for row in rows if row["generator"] == method]
        hl = np.array([row["hl_accuracy"] for row in subset]) * 100
        da = np.array([row["da_hl_accuracy"] for row in subset]) * 100
        print(f"{method:14s} HL {hl.mean():.2f} +/- {hl.std(ddof=1):.2f}; DA-HL {da.mean():.2f} +/- {da.std(ddof=1):.2f}; delta {np.mean(da-hl):+.2f}")


if __name__ == "__main__":
    main()
