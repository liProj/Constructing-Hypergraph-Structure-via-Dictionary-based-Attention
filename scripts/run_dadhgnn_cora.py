from __future__ import annotations

import argparse
import csv
from pathlib import Path

from da_hypergraph.citation import global_dictionary_attention, neighborhood_incidence, propagation_sparse
from da_hypergraph.data import load_planetoid
from da_hypergraph.models import train_network


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dictionary-iter", type=int, default=30)
    args = parser.parse_args()

    features, labels, adjacency, train, validation, test = load_planetoid(
        ROOT / "data" / "raw" / "planetoid", "cora"
    )
    h = neighborhood_incidence(adjacency)
    h_da = global_dictionary_attention(features, h, max_iter=args.dictionary_iter)
    theta = propagation_sparse(h)
    theta_da = propagation_sparse(h_da)
    rows = []
    for seed in range(args.repeats):
        baseline, baseline_epoch = train_network(
            features, labels, theta, train, validation, test,
            hidden=256, learning_rate=0.1, epochs=args.epochs, random_state=seed,
        )
        attended, attended_epoch = train_network(
            features, labels, theta_da, train, validation, test,
            hidden=256, learning_rate=0.1, epochs=args.epochs, random_state=seed,
        )
        row = {
            "seed": seed,
            "baseline_accuracy": baseline,
            "da_accuracy": attended,
            "delta": attended - baseline,
            "baseline_best_epoch": baseline_epoch,
            "da_best_epoch": attended_epoch,
            "protocol": "node_neighborhood_hyperedges_global_dictionary_floor_0.5",
        }
        rows.append(row)
        print(row)

    output = ROOT / "results" / "reproduced" / "cora_dadhgnn.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
