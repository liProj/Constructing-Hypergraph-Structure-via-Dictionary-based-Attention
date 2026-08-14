# Constructing Hypergraph Structure via Dictionary-based Attention

This repository provides a CPU-oriented implementation and audit of:

> Y. Sun et al., "Constructing Hypergraph Structure via Dictionary-based Attention," *Neural Processing Letters*, 2025. DOI: [10.1007/s11063-025-11779-1](https://doi.org/10.1007/s11063-025-11779-1).

It implements the paper's dictionary-based attention (DA) block, the closed-form
DA-HL objective, three implicit hypergraph generators, and a scalable two-layer
incidence-network variant for citation data. It also preserves the values read
from the paper separately from independently reproduced measurements.

## Repository contents

```text
.
|-- da_hypergraph/           # DA, hypergraph construction, data and models
|-- scripts/                 # Download, reproduction, verification and figures
|-- data/README.md           # Dataset links, caveats and redistribution notes
|-- results/
|   |-- reported/            # Values transcribed from the source paper
|   |-- reproduced/          # Measurements produced in this environment
|   `-- figures/             # Figures generated from reproduced CSV files
|-- manuscript/              # Paper-ready LaTeX and Chinese audit notes
|-- tests/                   # Numerical invariants
|-- CITATION.cff
|-- NOTICE.md
`-- requirements.txt
```

## Method summary

For each hyperedge, DA maps vertex features to sparse dictionary codes and uses
the centroid/member code inner product as a membership weight. The resulting
incidence matrix is used by either hypergraph label propagation (DA-HL) or a
two-step vertex-edge network.

The source paper omits several implementation choices. This repository exposes
them instead of hiding them:

- DA-HL uses per-hyperedge dictionaries and clips cosine-normalized code
  similarities to `[eps, 1]` so degree matrices remain valid.
- The citation pilot uses node-centred closed neighborhoods, one global shared
  dictionary for scalability, and a 0.5 attention floor.
- The representation hypergraph uses ridge self-representation followed by
  top-k coefficient selection; the source paper names a family of methods but
  does not identify an exact implementation or hyperparameters.

These choices make the code runnable, but they are not claimed to be the
authors' unreleased implementation.

## Results reported in the paper

The following values are verified against Figures 4--6 and Table 2.

| Dataset / method | Baseline (%) | DA (%) | Change (pp) |
|---|---:|---:|---:|
| ET-YaleB, k-NN | 79.2 | 80.9 | +1.7 |
| MNIST, clustering | 64.3 | 67.7 | +3.4 |
| RSSCN7, representation | 68.1 | 68.7 | +0.6 |
| Cora, DHGNN | 82.5 | 83.8 | +1.3 |
| Citeseer, DHGNN | 71.2 | 72.1 | +0.9 |
| Pubmed, DHGNN | 79.8 | 81.5 | +1.7 |

All published values are in `results/reported/`, including the robustness and
parameter ablations.

## Independent CPU pilot

The local pilot used five deterministic seeds. MNIST used 50 labelled samples
(five per class) and 100 balanced test samples in the induced 150-node graph.
This is a deliberately bounded pilot, not the paper's potentially full
70,000-node transductive graph.

| Dataset / generator | Baseline, mean ± SD (%) | DA, mean ± SD (%) | Change (pp) |
|---|---:|---:|---:|
| MNIST, k-NN | 60.40 ± 2.88 | 61.60 ± 1.14 | +1.20 |
| MNIST, clustering | 48.40 ± 7.67 | 47.00 ± 7.11 | -1.40 |
| MNIST, ridge representation | 36.60 ± 3.21 | 42.20 ± 4.71 | +5.60 |
| Cora, two-step incidence network | 81.92 ± 1.20 | 81.22 ± 1.20 | -0.70 |

The mixed result is scientifically useful: the proposed weighting helps two of
three MNIST constructions, but it does not establish a generator-independent
gain under this independent protocol. The Cora variant also does not reproduce
the paper's average improvement, despite one seed reaching 83.1%. Exact
replication requires the authors' processed arrays, sampled indices, precise
hypergraph generators, centroid rules, initialization, stopping criteria, and
DHGNN implementation.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.download_data --citations
python -m unittest discover -s tests -v
python -m scripts.verify_reported_results
python -m scripts.run_dahl_mnist --repeats 5 --max-iter 8
python -m scripts.run_dadhgnn_cora --repeats 5 --epochs 150 --dictionary-iter 5
python -m scripts.make_figures
```

Raw files are kept under `data/raw/` and ignored by Git. SHA-256 values printed
by the downloader make the local inputs auditable.

See `manuscript/paper_results.tex` for paper-ready experimental text. Following
the manuscript policy, this LaTeX section contains only the source-paper values
and does not include the independent pilot measurements.
