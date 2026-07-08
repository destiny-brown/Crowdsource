import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime
    pd = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    DataLoader = None
    Dataset = object

try:
    from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:  # pragma: no cover - handled at runtime
    accuracy_score = None
    classification_report = None
    precision_recall_fscore_support = None
    roc_auc_score = None
    train_test_split = None
    compute_class_weight = None

try:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError:  # pragma: no cover - handled at runtime
    AutoConfig = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    get_linear_schedule_with_warmup = None


POSITIVE_HINTS = {
    "1",
    "true",
    "yes",
    "ai",
    "generated",
    "suspicious",
    "flagged",
    "positive",
}


@dataclass
class TextExample:
    text: str
    label: int


class EncodedTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a binary text classifier for OCR/LaTeX audit text.")
    parser.add_argument("--data", required=True, help="CSV file containing labeled text examples.")
    parser.add_argument("--text-column", default="text", help="Column containing the input text.")
    parser.add_argument("--label-column", default="label", help="Column containing the binary label.")
    parser.add_argument("--positive-label", default=None, help="Value that should be treated as the positive class.")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Base checkpoint to fine-tune.")
    parser.add_argument("--output-dir", default="text_classifier_model", help="Directory for the trained model.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum token length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model-type", choices=["distilbert", "deberta-v3-small"], default="distilbert", help="Convenience preset for documentation only.")
    return parser.parse_args()


def require_dependencies():
    missing = []
    for module_name, module in [
        ("numpy", np),
        ("pandas", pd),
        ("torch", torch),
        ("scikit-learn", accuracy_score),
        ("transformers", AutoConfig),
    ]:
        if module is None:
            missing.append(module_name)

    if missing:
        raise RuntimeError(
            "Missing required packages for training: "
            + ", ".join(missing)
            + ". Install them with: pip install -r requirements.txt"
        )


def set_seed(seed):
    if torch is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def infer_label_mapping(raw_labels, positive_label=None):
    normalized = [normalize_value(value) for value in raw_labels]
    unique_values = list(dict.fromkeys(normalized))

    if len(set(unique_values)) != 2:
        raise ValueError(f"Expected exactly 2 unique labels, found {sorted(set(unique_values))}")

    if positive_label is not None:
        positive = normalize_value(positive_label)
        if positive not in unique_values:
            raise ValueError(f"Positive label {positive_label!r} was not found in the dataset labels.")
    else:
        if any(value in POSITIVE_HINTS for value in unique_values):
            positive = next(value for value in unique_values if value in POSITIVE_HINTS)
        elif any(value in {"0", "1"} for value in unique_values):
            positive = "1"
        else:
            raise ValueError(
                "Could not infer the positive label automatically. Re-run with --positive-label."
            )

    negative = next(value for value in unique_values if value != positive)
    label_to_id = {negative: 0, positive: 1}
    id_to_label = {0: negative, 1: positive}
    return label_to_id, id_to_label


def prepare_examples(frame, text_column, label_column, label_to_id):
    examples = []
    for _, row in frame.iterrows():
        text = str(row[text_column]).strip()
        label_value = normalize_value(row[label_column])
        if label_value not in label_to_id:
            raise ValueError(f"Unexpected label value {row[label_column]!r}")
        examples.append(TextExample(text=text, label=label_to_id[label_value]))
    return examples


def build_dataset(tokenizer, examples, max_length):
    texts = [example.text for example in examples]
    labels = [example.label for example in examples]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return EncodedTextDataset(encodings, labels)


def evaluate(model, dataloader, device):
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for evaluation.")
    model.eval()
    losses = []
    all_labels = []
    all_probs = []
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probabilities.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    predictions = [1 if probability >= 0.5 else 0 for probability in all_probs]
    accuracy = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = None

    return {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "classification_report": classification_report(all_labels, predictions, zero_division=0),
    }


def main():
    args = parse_args()
    require_dependencies()
    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    frame = pd.read_csv(data_path)
    if args.text_column not in frame.columns:
        raise ValueError(f"Text column {args.text_column!r} was not found in {data_path}")
    if args.label_column not in frame.columns:
        raise ValueError(f"Label column {args.label_column!r} was not found in {data_path}")

    frame = frame.dropna(subset=[args.text_column, args.label_column]).copy()
    if len(frame) < 10:
        raise ValueError("Need at least 10 labeled rows to train a useful classifier.")

    label_to_id, id_to_label = infer_label_mapping(frame[args.label_column].tolist(), args.positive_label)
    examples = prepare_examples(frame, args.text_column, args.label_column, label_to_id)

    texts = [example.text for example in examples]
    labels = [example.label for example in examples]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=args.validation_split,
        random_state=args.seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: id_to_label[0], 1: id_to_label[1]},
        label2id={id_to_label[0]: 0, id_to_label[1]: 1},
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    train_dataset = build_dataset(tokenizer, [TextExample(text, label) for text, label in zip(train_texts, train_labels)], args.max_length)
    val_dataset = build_dataset(tokenizer, [TextExample(text, label) for text, label in zip(val_texts, val_labels)], args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=np.array(train_labels))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=max(1, total_steps),
    )

    best_f1 = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            labels_batch = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        metrics = evaluate(model, val_loader, device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else None
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={metrics['loss']:.4f} | "
            f"val_f1={metrics['f1']:.4f} | val_auc={metrics['roc_auc']}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "label_to_id": label_to_id,
                        "id_to_label": {str(key): value for key, value in id_to_label.items()},
                        "positive_label": id_to_label[1],
                        "negative_label": id_to_label[0],
                        "base_model": args.model_name,
                    },
                    handle,
                    indent=2,
                )

    final_metrics = evaluate(model, val_loader, device)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    print("Training complete.")
    print(f"Saved model to: {output_dir}")
    print(final_metrics["classification_report"])


if __name__ == "__main__":
    main()