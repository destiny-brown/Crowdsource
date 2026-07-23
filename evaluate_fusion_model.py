import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from train_fusion_model import build_features


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 fusion model with calibration + bootstrap CI")
    parser.add_argument("--model", required=True, help="Path to fusion_model.pkl")
    parser.add_argument("--data", required=True, help="CSV with labels and feature columns")
    parser.add_argument("--label-column", default="label", help="Binary label column")
    parser.add_argument("--positive-label", default="1", help="Value treated as positive class")
    parser.add_argument("--out", default="fusion_model_eval.json", help="Path to save evaluation JSON")
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def percentile_interval(values, low=2.5, high=97.5):
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def safe_metric(func, y_true, y_score_or_pred):
    try:
        return float(func(y_true, y_score_or_pred))
    except Exception:
        return None


def main():
    args = parse_args()

    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies with: pip install scikit-learn") from exc

    model_path = Path(args.model)
    with open(model_path, "rb") as handle:
        artifact = pickle.load(handle)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    fill_values = artifact["fill_values"]

    frame = pd.read_csv(args.data)
    if args.label_column not in frame.columns:
        raise ValueError(f"Missing label column: {args.label_column}")

    labels_raw = frame[args.label_column].astype(str).str.strip().str.lower()
    positive = str(args.positive_label).strip().lower()
    y = (labels_raw == positive).astype("int64").values

    features, _ = build_features(frame)
    for column in feature_columns:
        if column not in features.columns:
            features[column] = float(fill_values.get(column, 0.0))
    X = features[feature_columns].fillna(value=fill_values).values.astype("float32")

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype("int64")

    metrics = {
        "rows": int(len(frame)),
        "auc": safe_metric(roc_auc_score, y, probs),
        "brier": safe_metric(brier_score_loss, y, probs),
        "accuracy": safe_metric(accuracy_score, y, preds),
        "precision": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y, preds),
        "recall": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y, preds),
        "f1": safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y, preds),
        "threshold": 0.5,
    }

    prob_true, prob_pred = calibration_curve(y, probs, n_bins=min(10, max(2, len(frame) // 2)), strategy="uniform")
    calibration = [
        {"pred_mean": float(p), "true_rate": float(t)}
        for p, t in zip(prob_pred, prob_true)
    ]

    rng = np.random.default_rng(args.seed)
    boot = {"auc": [], "brier": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    n = len(y)
    for _ in range(max(1, args.bootstrap)):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        p_b = probs[idx]
        pred_b = (p_b >= 0.5).astype("int64")

        auc_b = None
        if len(np.unique(y_b)) > 1:
            auc_b = safe_metric(roc_auc_score, y_b, p_b)
        if auc_b is not None:
            boot["auc"].append(auc_b)
        brier_b = safe_metric(brier_score_loss, y_b, p_b)
        if brier_b is not None:
            boot["brier"].append(brier_b)
        boot["accuracy"].append(safe_metric(accuracy_score, y_b, pred_b) or 0.0)
        boot["precision"].append(safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_b, pred_b) or 0.0)
        boot["recall"].append(safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_b, pred_b) or 0.0)
        boot["f1"].append(safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_b, pred_b) or 0.0)

    ci = {}
    for key, values in boot.items():
        if values:
            lo, hi = percentile_interval(values)
            ci[key] = {"low": lo, "high": hi}
        else:
            ci[key] = {"low": None, "high": None}

    result = {
        "model_path": str(model_path),
        "artifact_version": artifact.get("artifact_version", "unknown"),
        "trained_at_utc": artifact.get("trained_at_utc", "unknown"),
        "metrics": metrics,
        "metrics_ci_95": ci,
        "calibration_curve": calibration,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved evaluation: {out_path}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
