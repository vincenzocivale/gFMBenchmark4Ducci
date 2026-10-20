"""Content-addressed representations with sequence-level window aggregation."""

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import __version__
from .data import read_records, validate
from .encoders import create_encoder


def fingerprint(records, model_config, window_bp):
    payload = {"version": __version__, "records": [asdict(r) for r in records],
               "encoder": model_config, "window_bp": window_bp}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract(records, encoder, window_bp=512, batch_size=8):
    if window_bp < 1 or batch_size < 1:
        raise ValueError("window_bp and batch_size must be positive")
    result = []
    for record in records:
        total, weight = None, 0
        for start in range(0, len(record.sequence), window_bp * batch_size):
            region = record.sequence[start:start + window_bp * batch_size]
            windows = [region[i:i + window_bp] for i in range(0, len(region), window_bp)]
            vectors = np.asarray(encoder.encode(windows), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(windows) or not np.isfinite(vectors).all():
                raise ValueError("Encoder must return finite [n_windows, n_features] matrix")
            weights = np.array([len(w) for w in windows])
            subtotal = (vectors * weights[:, None]).sum(axis=0)
            total = subtotal if total is None else total + subtotal
            weight += weights.sum()
        result.append(total / weight)
    return np.stack(result).astype(np.float32)


def run(config, config_dir=Path("."), selected=None, embeddings_only=False):
    if not embeddings_only:
        from .evaluation import evaluate
    if config.get("task", "classification") not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    dataset = config["dataset"]
    records = list(read_records(config_dir / dataset["path"], dataset.get("columns"), dataset.get("limit")))
    validate(records, supervised=not embeddings_only)
    window_bp = config.get("window_bp", 512)
    output = config_dir / config.get("output_dir", "outputs")
    output.mkdir(parents=True, exist_ok=True)
    models = config["models"]
    if selected:
        unknown = set(selected) - models.keys()
        if unknown:
            raise ValueError(f"Unknown model names: {sorted(unknown)}")
        models = {name: models[name] for name in selected}
    summaries = {}
    for name, model_config in models.items():
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Model names must be simple alphanumeric identifiers")
        key = fingerprint(records, model_config, window_bp)
        directory = output / name / key
        directory.mkdir(parents=True, exist_ok=True)
        feature_path = directory / "features.npy"
        if feature_path.exists():
            features = np.load(feature_path, allow_pickle=False)
        else:
            print(f"Extracting {name}: {len(records)} sequences", flush=True)
            encoder = create_encoder(model_config)
            # K-mer frequencies describe the complete input, including window boundaries.
            features = (encoder.encode([r.sequence for r in records]) if model_config["backend"] == "kmer"
                        else extract(records, encoder, window_bp, config.get("batch_size", 8)))
            np.save(feature_path, features, allow_pickle=False)
            del encoder
        if features.ndim != 2 or features.shape[0] != len(records) or not np.isfinite(features).all():
            raise ValueError("Invalid cached feature matrix")
        packages = {}
        for package in ("numpy", "scikit-learn", "torch", "transformers", "jax", "evo2"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        manifest = {"fingerprint": key, "config": config, "model": name,
                    "ids": [r.id for r in records], "shape": list(features.shape),
                    "python": platform.python_version(), "packages": packages,
                    "benchmark_version": __version__}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if not embeddings_only:
            metrics, predictions = evaluate(features, records, config.get("task", "classification"), config.get("seed", 42))
            (directory / "metrics.json").write_text(json.dumps(metrics, indent=2))
            test_records = [r for r in records if r.split == "test"]
            with (directory / "predictions.jsonl").open("w") as handle:
                for record, pred in zip(test_records, predictions):
                    handle.write(json.dumps({"id": record.id, "label": record.label, "prediction": pred.item()}) + "\n")
            summaries[name] = metrics
            print(f"{name}: {metrics['test']}", flush=True)
        else:
            print(f"{name}: saved {features.shape} to {directory}", flush=True)
    if summaries:
        report_key = hashlib.sha256(json.dumps({"config": config, "models": list(models)}, sort_keys=True).encode()).hexdigest()[:12]
        (output / f"summary-{report_key}.json").write_text(json.dumps(summaries, indent=2))
    return summaries
