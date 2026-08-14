from pathlib import Path

import argparse

from da_hypergraph.data import download_mnist, download_planetoid, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--citations", action="store_true", help="also download Cora, Citeseer and Pubmed")
    args = parser.parse_args()
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    paths = download_mnist(raw / "mnist")
    if args.citations:
        for name in ("cora", "citeseer", "pubmed"):
            paths.extend(download_planetoid(raw / "planetoid", name))
    for path in paths:
        print(f"{path.name}\t{path.stat().st_size}\tsha256={sha256(path)}")


if __name__ == "__main__":
    main()
