"""Train-only scaling, validation-only model selection, one final test."""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(features, records, task="classification", seed=42, strengths=(0.01, 0.1, 1, 10, 100)):
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    splits = np.array([r.split for r in records])
    y = np.array([r.label for r in records], dtype=float if task == "regression" else str)
    masks = {s: splits == s for s in ("train", "validation", "test")}
    if task == "regression" and (not np.isfinite(y).all() or any(m.sum() < 2 for m in masks.values())):
        raise ValueError("Regression needs finite targets and at least two records per split")
    if task == "classification":
        classes = set(y[masks["train"]])
        if len(classes) < 2 or any(set(y[m]) != classes for m in masks.values()):
            raise ValueError("Every split must contain all training classes (at least two)")
    best = None
    for strength in strengths:
        estimator = (LogisticRegression(C=strength, max_iter=2000, random_state=seed)
                     if task == "classification" else Ridge(alpha=strength))
        probe = make_pipeline(StandardScaler(), estimator)
        probe.fit(features[masks["train"]], y[masks["train"]])
        pred = probe.predict(features[masks["validation"]])
        score = (balanced_accuracy_score(y[masks["validation"]], pred) if task == "classification"
                 else -mean_squared_error(y[masks["validation"]], pred))
        if best is None or score > best[0]:
            best = score, strength, probe
    score, strength, probe = best
    truth = y[masks["test"]]
    test_x = features[masks["test"]]
    prediction = probe.predict(test_x)
    if task == "classification":
        metrics = dict(accuracy=accuracy_score(truth, prediction),
                       balanced_accuracy=balanced_accuracy_score(truth, prediction),
                       f1_macro=f1_score(truth, prediction, average="macro", zero_division=0),
                       mcc=matthews_corrcoef(truth, prediction))
        probabilities = probe.predict_proba(test_x)
        if probabilities.shape[1] == 2:
            binary = truth == probe.classes_[1]
            metrics.update(auroc=roc_auc_score(binary, probabilities[:, 1]),
                           average_precision=average_precision_score(binary, probabilities[:, 1]))
    else:
        metrics = dict(mae=mean_absolute_error(truth, prediction),
                       rmse=float(np.sqrt(mean_squared_error(truth, prediction))),
                       r2=r2_score(truth, prediction))
    return {"task": task, "strength": strength, "validation_score": float(score),
            "test": {k: float(v) for k, v in metrics.items()},
            "n": {s: int(m.sum()) for s, m in masks.items()}}, prediction
