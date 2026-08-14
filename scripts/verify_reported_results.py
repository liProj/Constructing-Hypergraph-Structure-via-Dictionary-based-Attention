import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ("ET-YaleB", "knn"): (79.2, 80.9),
    ("ET-YaleB", "clustering"): (77.4, 80.4),
    ("ET-YaleB", "representation"): (80.3, 81.1),
    ("MNIST", "knn"): (67.2, 69.5),
    ("MNIST", "clustering"): (64.3, 67.7),
    ("MNIST", "representation"): (68.4, 69.9),
    ("RSSCN7", "knn"): (67.5, 68.6),
    ("RSSCN7", "clustering"): (66.4, 68.1),
    ("RSSCN7", "representation"): (68.1, 68.7),
}


def main() -> None:
    path = ROOT / "results" / "reported" / "visual_results.csv"
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    actual = {(r["dataset"], r["generator"]): (float(r["hl_accuracy_pct"]), float(r["da_hl_accuracy_pct"])) for r in rows}
    if actual != EXPECTED:
        raise SystemExit(f"reported values differ from the source paper: {actual}")
    print("Reported visual-recognition values match Figures 4--6 of the source paper.")


if __name__ == "__main__":
    main()
