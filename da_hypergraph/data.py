from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path
import pickle
import time

import numpy as np
import scipy.sparse as sp


MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

PLANETOID_OBJECTS = ("x", "tx", "allx", "y", "ty", "ally", "graph")
PLANETOID_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"


def _download_url(url: str, destination: Path, retries: int = 4) -> None:
    expected = None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            expected = int(response.headers.get("Content-Length", "0")) or None
    except Exception:
        pass
    if destination.exists() and destination.stat().st_size > 0:
        if expected is None or destination.stat().st_size == expected:
            return
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, temporary)
            if expected is not None and temporary.stat().st_size != expected:
                raise IOError(f"expected {expected} bytes, got {temporary.stat().st_size}")
            temporary.replace(destination)
            return
        except Exception as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_mnist(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in MNIST_FILES.items():
        destination = root / name
        _download_url(url, destination)
        paths.append(destination)
    return paths


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        zero, dtype_code, dimensions = struct.unpack(">HBB", stream.read(4))
        if zero != 0 or dtype_code != 8:
            raise ValueError(f"unsupported IDX header in {path}")
        shape = struct.unpack(">" + "I" * dimensions, stream.read(4 * dimensions))
        return np.frombuffer(stream.read(), dtype=np.uint8).reshape(shape)


def load_mnist(root: Path, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if download:
        download_mnist(root)
    train_x = _read_idx(root / "train-images-idx3-ubyte.gz")
    train_y = _read_idx(root / "train-labels-idx1-ubyte.gz")
    test_x = _read_idx(root / "t10k-images-idx3-ubyte.gz")
    test_y = _read_idx(root / "t10k-labels-idx1-ubyte.gz")
    x = np.concatenate([train_x, test_x]).reshape(-1, 784).astype(np.float64) / 255.0
    y = np.concatenate([train_y, test_y]).astype(int)
    return x, y


def balanced_paper_split(
    labels: np.ndarray,
    *,
    train_per_class: int = 5,
    test_total: int = 100,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    classes = np.unique(labels)
    if test_total % len(classes):
        raise ValueError("test_total must be divisible by the number of classes")
    train, test = [], []
    test_per_class = test_total // len(classes)
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        chosen = rng.choice(candidates, size=train_per_class + test_per_class, replace=False)
        train.extend(chosen[:train_per_class])
        test.extend(chosen[train_per_class:])
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def download_planetoid(root: Path, name: str) -> list[Path]:
    if name not in {"cora", "citeseer", "pubmed"}:
        raise ValueError(name)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (*PLANETOID_OBJECTS, "test.index"):
        filename = f"ind.{name}.{suffix}"
        destination = root / filename
        _download_url(f"{PLANETOID_BASE}/{filename}", destination)
        paths.append(destination)
    return paths


def load_planetoid(root: Path, name: str, download: bool = True):
    """Load a Planetoid citation network and its standard fixed split."""
    if download:
        download_planetoid(root, name)
    objects = []
    for suffix in PLANETOID_OBJECTS:
        with (root / f"ind.{name}.{suffix}").open("rb") as stream:
            objects.append(pickle.load(stream, encoding="latin1"))
    x, tx, allx, y, ty, ally, graph = objects
    test_reorder = np.loadtxt(root / f"ind.{name}.test.index", dtype=int)
    test_range = np.sort(test_reorder)

    if name == "citeseer":
        full = range(min(test_reorder), max(test_reorder) + 1)
        tx_extended = sp.lil_matrix((len(full), x.shape[1]))
        tx_extended[test_range - min(test_reorder), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(full), y.shape[1]))
        ty_extended[test_range - min(test_reorder), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_reorder, :] = features[test_range, :]
    labels_onehot = np.vstack((ally, ty))
    labels_onehot[test_reorder, :] = labels_onehot[test_range, :]
    labels = labels_onehot.argmax(axis=1).astype(int)

    rows, cols = [], []
    for source, targets in graph.items():
        rows.extend([source] * len(targets))
        cols.extend(targets)
    adjacency = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float64,
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    row_sum = np.asarray(features.sum(axis=1)).ravel()
    features = sp.diags(1.0 / np.maximum(row_sum, 1e-12)) @ features.tocsr()

    train = np.arange(len(y), dtype=int)
    validation = np.arange(len(y), len(y) + 500, dtype=int)
    test = test_range.astype(int)
    return features, labels, adjacency, train, validation, test
