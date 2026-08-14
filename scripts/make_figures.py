from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    mnist = pd.read_csv(ROOT / "results" / "reproduced" / "mnist_dahl.csv")
    cora = pd.read_csv(ROOT / "results" / "reproduced" / "cora_dadhgnn.csv")
    output = ROOT / "results" / "figures"
    output.mkdir(parents=True, exist_ok=True)

    summary = mnist.groupby("generator")[["hl_accuracy", "da_hl_accuracy"]].agg(["mean", "std"])
    order = ["knn", "clustering", "representation"]
    x = range(len(order))
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar([i - 0.18 for i in x], [100 * summary.loc[i, ("hl_accuracy", "mean")] for i in order],
             0.36, yerr=[100 * summary.loc[i, ("hl_accuracy", "std")] for i in order], label="HL")
    axis.bar([i + 0.18 for i in x], [100 * summary.loc[i, ("da_hl_accuracy", "mean")] for i in order],
             0.36, yerr=[100 * summary.loc[i, ("da_hl_accuracy", "std")] for i in order], label="DA-HL")
    axis.set_xticks(list(x), ["k-NN", "Clustering", "Ridge representation"])
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("MNIST pilot reproduction (5 seeds; mean ± SD)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "mnist_pilot.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    means = [100 * cora.baseline_accuracy.mean(), 100 * cora.da_accuracy.mean()]
    errors = [100 * cora.baseline_accuracy.std(), 100 * cora.da_accuracy.std()]
    axis.bar([0, 1], means, yerr=errors, width=0.55, color=["#4c78a8", "#e45756"])
    axis.set_xticks([0, 1], ["Baseline", "DA variant"])
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Cora pilot reproduction (5 seeds; mean ± SD)")
    axis.set_ylim(max(0, min(means) - 4), min(100, max(means) + 4))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "cora_pilot.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
