import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit_report import TECHNICAL_REPORT_FILENAME


def parse_args():
    parser = argparse.ArgumentParser(description="Train calibrated Phase 2 fusion model")
    parser.add_argument("--data", default=TECHNICAL_REPORT_FILENAME, help="CSV file with engineered feature columns")
    parser.add_argument("--label-column", default="label", help="Binary label column")
    parser.add_argument("--positive-label", default="1", help="Value treated as positive class")
    parser.add_argument("--model-out", default="fusion_model.pkl", help="Path to output model pickle")
    parser.add_argument("--metrics-out", default="fusion_model_metrics.json", help="Path to metrics JSON")
    parser.add_argument("--test-size", type=float, default=0.25, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def to_float(series):
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({"N/A": np.nan, "": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def to_binary(series):
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "on", "y"}).astype("float32")


def duplicate_to_binary(series):
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.str.startswith("yes").astype("float32")


def build_features(frame):
    features = pd.DataFrame()
    features["ai_keyword_score"] = to_float(frame.get("AI Keyword Score", np.nan))
    features["text_classifier_score"] = to_float(frame.get("Text Classifier Score", np.nan))
    features["image_anomaly_score"] = to_float(frame.get("Image Anomaly Score", np.nan))
    features["forgery_cue_score"] = to_float(frame.get("Forgery Cue Score", np.nan))
    features["duration_risk_score"] = to_float(frame.get("Duration Risk Score", np.nan))
    features["metadata_missing"] = to_binary(frame.get("Metadata Missing", "no"))
    features["metadata_corroborated"] = to_binary(frame.get("Metadata Corroborated", "no"))
    features["is_duplicate"] = duplicate_to_binary(frame.get("Is Duplicate", "no"))

    security_flags = frame.get("Security Flags", "").astype(str)
    features["id_mismatch"] = security_flags.str.contains(
        r"ID mismatch \(Participant not found in survey CSV export\)", regex=True
    ).astype("float32")

    fill_values = features.median(numeric_only=True).fillna(0.0).to_dict()
    features = features.fillna(value=fill_values)
    return features.astype("float32"), fill_values


def main():
    args = parse_args()

    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install scikit-learn") from exc

    frame = pd.read_csv(args.data)
    if args.label_column not in frame.columns:
        raise ValueError(f"Missing label column: {args.label_column}")

    labels_raw = frame[args.label_column].astype(str).str.strip().str.lower()
    positive = str(args.positive_label).strip().lower()
    labels = (labels_raw == positive).astype("int64")
    if labels.nunique() < 2:
        raise ValueError("Need both positive and negative labels for training")

    features, fill_values = build_features(frame)
    feature_columns = list(features.columns)

    label_counts = labels.value_counts()
    min_class_count = int(label_counts.min())
    total_rows = int(len(frame))

    base_model = LogisticRegression(max_iter=2000, class_weight="balanced")

    use_small_data_mode = total_rows < 20 or min_class_count < 4
    if use_small_data_mode:
        model = base_model
        model.fit(features.values, labels.values)
        probs = model.predict_proba(features.values)[:, 1]
        preds = (probs >= 0.5).astype("int64")
        eval_labels = labels.values
        eval_split = "train_only"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features.values,
            labels.values,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels.values,
        )

        model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype("int64")
        eval_labels = y_test
        eval_split = "holdout"

    metrics = {
        "rows": int(len(frame)),
        "feature_columns": feature_columns,
        "evaluation_split": eval_split,
        "small_data_mode": use_small_data_mode,
        "auc": float(roc_auc_score(eval_labels, probs)),
        "brier": float(brier_score_loss(eval_labels, probs)),
        "accuracy": float(accuracy_score(eval_labels, preds)),
        "precision": float(precision_score(eval_labels, preds, zero_division=0)),
        "recall": float(recall_score(eval_labels, preds, zero_division=0)),
        "f1": float(f1_score(eval_labels, preds, zero_division=0)),
        "threshold": 0.5,
    }

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "fill_values": fill_values,
        "metrics": metrics,
        "artifact_version": "phase2-v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    with open(model_path, "wb") as handle:
        pickle.dump(artifact, handle)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
