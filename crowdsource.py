import os
import io
import hashlib
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
from PIL import Image, ImageChops, ImageFilter
from PIL.ExifTags import TAGS
from math_ocr_api import convert_math_photo_to_latex, load_project_env
from audit_report import (
    build_review_row,
    classify_risk_tier,
    format_report_value,
    save_audit_reports,
)


# --- CONFIGURATION ---
load_project_env()

MINIMUM_MINUTES_REQUIRED = 10.0  # Change to survey's minimum threshold
SURVEY_CSV_PATH = "survey_export.csv"
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "").strip()
DROPBOX_SOURCE_FOLDER = os.getenv("DROPBOX_SOURCE_FOLDER", "/intake/raw").strip()
ENABLE_IMAGE_ANOMALY_SCORING = os.getenv("ENABLE_IMAGE_ANOMALY_SCORING", "false").strip().lower() in {"1", "true", "yes", "on"}
IMAGE_EMBEDDING_BACKBONE = os.getenv("IMAGE_EMBEDDING_BACKBONE", "open_clip").strip().lower()
IMAGE_ANOMALY_DETECTOR = os.getenv("IMAGE_ANOMALY_DETECTOR", "isolation_forest").strip().lower()
IMAGE_ANOMALY_CONTAMINATION = float(os.getenv("IMAGE_ANOMALY_CONTAMINATION", "0.1"))
IMAGE_ANOMALY_FLAG_THRESHOLD = float(os.getenv("IMAGE_ANOMALY_FLAG_THRESHOLD", "70.0"))
TEXT_CLASSIFIER_MODEL = os.getenv("TEXT_CLASSIFIER_MODEL")
ENABLE_TEXT_CLASSIFIER = os.getenv("ENABLE_TEXT_CLASSIFIER", "false").strip().lower() in {"1", "true", "yes", "on"} or bool(TEXT_CLASSIFIER_MODEL)
TEXT_CLASSIFIER_THRESHOLD = float(os.getenv("TEXT_CLASSIFIER_THRESHOLD", "0.7"))
TEXT_CLASSIFIER_THRESHOLD_PERCENT = TEXT_CLASSIFIER_THRESHOLD * 100.0
FUSION_WEIGHT_TEXT = float(os.getenv("FUSION_WEIGHT_TEXT", "0.30"))
FUSION_WEIGHT_FORGERY = float(os.getenv("FUSION_WEIGHT_FORGERY", "0.25"))
FUSION_WEIGHT_ANOMALY = float(os.getenv("FUSION_WEIGHT_ANOMALY", "0.20"))
FUSION_WEIGHT_DURATION = float(os.getenv("FUSION_WEIGHT_DURATION", "0.10"))
FUSION_WEIGHT_METADATA = float(os.getenv("FUSION_WEIGHT_METADATA", "0.10"))
FUSION_WEIGHT_DUPLICATE = float(os.getenv("FUSION_WEIGHT_DUPLICATE", "0.05"))
FUSION_MODEL_PATH = os.getenv("FUSION_MODEL_PATH", "").strip()

AI_KEYWORD_FLAG_LIST = {
    "Overly Academic Adjectives": [
        "meticulous",
        "comprehensive",
        "intricate",
        "straightforward",
        "rigorous",
        "profound",
        "elegant"
    ],
    "Step-by-Step Setup": [
        "let's break this down",
        "to solve this step-by-step",
        "crucially",
        "importantly",
        "let's delve into",
        "it's worth noting"
    ],
    "Over-Explainer Transitions": [
        "consequently",
        "hence",
        "thus",
        "therefore, we can conclude",
        "by applying the principles of",
        "as a result"
    ],
    "Wrapping Up Fluff": [
        "in summary",
        "ultimately",
        "this gives us the final result of",
        "we successfully determined"
    ]
}
AI_KEYWORD_SCORE_MATCHES_FOR_MAX = 6


def get_participant_id(filename):
    return os.path.splitext(filename)[0] if filename else ""


# --- DROPBOX ACCESS ---
def get_dropbox_client():
    if not DROPBOX_ACCESS_TOKEN:
        raise ValueError("Set DROPBOX_ACCESS_TOKEN for Dropbox API access.")

    try:
        import dropbox
    except ImportError as exc:
        raise ImportError("Install Dropbox dependency with: pip install dropbox") from exc

    client = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    client.users_get_current_account()  # Validate token
    return client


def load_dropbox_images(folder_path=DROPBOX_SOURCE_FOLDER):
    if not folder_path.startswith("/"):
        folder_path = f"/{folder_path}"

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    client = get_dropbox_client()

    items = []
    response = client.files_list_folder(folder_path, recursive=False)

    while True:
        for entry in response.entries:
            if not hasattr(entry, "path_lower") or not hasattr(entry, "name"):
                continue

            _, ext = os.path.splitext(entry.name)
            if ext.lower() not in supported_extensions:
                continue

            _, file_response = client.files_download(entry.path_lower)
            items.append({
                "name": entry.name,
                "bytes": file_response.content,
            })

        if not response.has_more:
            break
        response = client.files_list_folder_continue(response.cursor)

    if not items:
        raise ValueError(f"No supported image files found in Dropbox folder: {folder_path}")

    return items


# --- OCR AND AI ANALYSIS ---


def clamp_score(value):
    return max(0.0, min(100.0, float(value)))


def compute_image_forgery_cues(image_bytes):
    empty_forgery_result = {
        "ela_score": None,
        "jpeg_recompression_score": None,
        "noise_residual_score": None,
        "blur_risk_score": None,
        "forgery_cue_score": None,
        "forgery_cue_flags": [],
    }
    if not image_bytes:
        return empty_forgery_result

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ELA-style cue: large reconstruction deltas can indicate edited regions.
        ela_buffer = io.BytesIO()
        image.save(ela_buffer, format="JPEG", quality=90)
        ela_recompressed = Image.open(io.BytesIO(ela_buffer.getvalue())).convert("RGB")
        ela_delta = np.abs(
            np.asarray(ImageChops.difference(image, ela_recompressed), dtype=np.float32)
        )
        ela_mean = float(np.mean(ela_delta))
        ela_score = clamp_score(ela_mean * 3.0)

        high_quality_buffer = io.BytesIO()
        low_quality_buffer = io.BytesIO()
        image.save(high_quality_buffer, format="JPEG", quality=95)
        image.save(low_quality_buffer, format="JPEG", quality=75)
        pixels_at_quality_95 = np.asarray(
            Image.open(io.BytesIO(high_quality_buffer.getvalue())).convert("RGB"),
            dtype=np.float32,
        )
        pixels_at_quality_75 = np.asarray(
            Image.open(io.BytesIO(low_quality_buffer.getvalue())).convert("RGB"),
            dtype=np.float32,
        )
        jpeg_gap = float(np.mean(np.abs(pixels_at_quality_95 - pixels_at_quality_75)))
        jpeg_recompression_score = clamp_score(jpeg_gap * 2.5)

        gray = image.convert("L")
        gray_np = np.asarray(gray, dtype=np.float32)
        blurred_np = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32)
        residual = gray_np - blurred_np
        residual_std = float(np.std(residual))
        noise_residual_score = clamp_score(residual_std * 4.0)

        gradient_y, gradient_x = np.gradient(gray_np)
        gradient_mag = np.sqrt((gradient_x * gradient_x) + (gradient_y * gradient_y))
        edge_strength = float(np.mean(gradient_mag))
        blur_risk_score = clamp_score(((18.0 - edge_strength) / 18.0) * 100.0)

        forgery_cue_score = round(
            (0.40 * ela_score)
            + (0.25 * jpeg_recompression_score)
            + (0.20 * noise_residual_score)
            + (0.15 * blur_risk_score),
            1,
        )

        forgery_flags = []
        if ela_score >= 70:
            forgery_flags.append("High ELA delta")
        if jpeg_recompression_score >= 70:
            forgery_flags.append("High recompression instability")
        if noise_residual_score >= 70:
            forgery_flags.append("Unusual residual noise")
        if blur_risk_score >= 70:
            forgery_flags.append("Low edge sharpness")

        return {
            "ela_score": round(ela_score, 1),
            "jpeg_recompression_score": round(jpeg_recompression_score, 1),
            "noise_residual_score": round(noise_residual_score, 1),
            "blur_risk_score": round(blur_risk_score, 1),
            "forgery_cue_score": forgery_cue_score,
            "forgery_cue_flags": forgery_flags,
        }
    except Exception as exc:
        print(f"Image forgery cue extraction failed: {exc}")
        return empty_forgery_result


def compute_duration_risk(duration_minutes):
    if duration_minutes is None:
        return None
    shortfall = max(0.0, MINIMUM_MINUTES_REQUIRED - float(duration_minutes))
    if MINIMUM_MINUTES_REQUIRED <= 0:
        return None
    return round(clamp_score((shortfall / MINIMUM_MINUTES_REQUIRED) * 100.0), 1)


def compute_fusion_risk(
    ai_keyword_score,
    text_classifier_score,
    image_anomaly_score,
    forgery_cue_score,
    duration_risk,
    is_metadata_missing,
    is_duplicate,
    is_id_mismatch,
):
    text_score = text_classifier_score if text_classifier_score is not None else ai_keyword_score
    metadata_risk = 70.0 if is_metadata_missing else 0.0
    duplicate_risk = 100.0 if is_duplicate else 0.0
    id_mismatch_risk = 100.0 if is_id_mismatch else 0.0
    label_map = {
        "text": "Text",
        "forgery": "Forgery",
        "anomaly": "ImageAnomaly",
        "duration": "Duration",
        "metadata": "Metadata",
        "duplicate": "Duplicate",
        "id_mismatch": "IDMismatch",
    }

    weighted_components = []
    base_weights = {
        "text": FUSION_WEIGHT_TEXT,
        "forgery": FUSION_WEIGHT_FORGERY,
        "anomaly": FUSION_WEIGHT_ANOMALY,
        "duration": FUSION_WEIGHT_DURATION,
        "metadata": FUSION_WEIGHT_METADATA,
        "duplicate": FUSION_WEIGHT_DUPLICATE,
        "id_mismatch": 0.10,
    }
    candidate_components = {
        "text": text_score,
        "forgery": forgery_cue_score,
        "anomaly": image_anomaly_score,
        "duration": duration_risk,
        "metadata": metadata_risk,
        "duplicate": duplicate_risk,
        "id_mismatch": id_mismatch_risk,
    }

    for name, value in candidate_components.items():
        if value is None:
            continue
        score = clamp_score(value)
        weighted_components.append(
            {
                "name": name,
                "label": label_map[name],
                "weight": base_weights[name],
                "score": score,
                "weighted": base_weights[name] * score,
            }
        )

    if not weighted_components:
        return {
            "score": None,
            "confidence": None,
            "tier": "Unknown",
            "why": "No components available",
            "why_top": "None",
        }

    weight_total = sum(component["weight"] for component in weighted_components)
    normalized_score = sum(component["weighted"] for component in weighted_components) / weight_total
    component_values = [component["score"] for component in weighted_components]

    coverage = len(component_values) / len(candidate_components)
    dispersion = float(np.std(component_values)) if len(component_values) > 1 else 0.0
    agreement = clamp_score(100.0 - (dispersion * 1.5))
    confidence = round((coverage * 100.0 * 0.6) + (agreement * 0.4), 1)

    score = round(clamp_score(normalized_score), 1)
    tier = classify_risk_tier(score)

    sorted_components = sorted(weighted_components, key=lambda component: component["weighted"], reverse=True)
    top_components = sorted_components[:3]
    why_top = "; ".join(
        f"{component['label']}({component['score']:.1f}*{component['weight']:.2f})"
        for component in top_components
    )
    why = (
        f"Top={why_top} | Coverage={len(component_values)}/{len(candidate_components)} "
        f"Agreement={agreement:.1f}"
    )

    return {
        "score": score,
        "confidence": confidence,
        "tier": tier,
        "why": why,
        "why_top": why_top,
    }


@lru_cache(maxsize=1)
def load_phase2_fusion_model():
    if not FUSION_MODEL_PATH or not os.path.isfile(FUSION_MODEL_PATH):
        return None
    try:
        with open(FUSION_MODEL_PATH, "rb") as model_file:
            model_artifact = pickle.load(model_file)
    except Exception as exc:
        print(f"Could not load fusion model at {FUSION_MODEL_PATH}: {exc}")
        return None

    if not isinstance(model_artifact, dict):
        print("Fusion model artifact is invalid (expected dict).")
        return None

    required_keys = {"model", "feature_columns", "fill_values"}
    if not required_keys.issubset(set(model_artifact.keys())):
        print("Fusion model artifact missing required keys.")
        return None
    return model_artifact


def score_phase2_fusion_model(feature_values):
    model_artifact = load_phase2_fusion_model()
    if model_artifact is None:
        return None

    try:
        model = model_artifact["model"]
        feature_columns = model_artifact["feature_columns"]
        fill_values = model_artifact["fill_values"]
        training_metrics = model_artifact.get("metrics", {})
        feature_row = []
        for column in feature_columns:
            value = feature_values.get(column)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = fill_values.get(column, 0.0)
            feature_row.append(float(value))

        probability = float(model.predict_proba(np.array([feature_row], dtype="float32"))[0, 1])
        score = round(clamp_score(probability * 100.0), 1)
        # Approximate 95% CI using binomial standard error and training row count.
        effective_sample_size = max(1, int(training_metrics.get("rows", 1)))
        std_err = float(np.sqrt(max(1e-9, probability * (1.0 - probability) / effective_sample_size)))
        ci_low = round(clamp_score((probability - (1.96 * std_err)) * 100.0), 1)
        ci_high = round(clamp_score((probability + (1.96 * std_err)) * 100.0), 1)
        confidence = round(clamp_score(abs(probability - 0.5) * 200.0), 1)
        tier = classify_risk_tier(score)

        return {
            "score": score,
            "confidence": confidence,
            "tier": tier,
            "why": f"Phase2 calibrated model probability={probability:.3f}",
            "why_top": "Calibrated model output",
            "ci_low": ci_low,
            "ci_high": ci_high,
            "model_version": model_artifact.get("artifact_version", "unknown"),
            "model_trained_at": model_artifact.get("trained_at_utc", "unknown"),
            "model_training_rows": training_metrics.get("rows", "unknown"),
        }
    except Exception as exc:
        print(f"Phase2 fusion model scoring failed: {exc}")
        return None


def score_ai_keyword_flags(text):
    normalized_text = (text or "").strip()
    if not normalized_text:
        return 0.0, []
    normalized_text = normalized_text.casefold()
    matched_keywords = []
    total_matches = 0
    for category, keywords in AI_KEYWORD_FLAG_LIST.items():
        for keyword in keywords:
            occurrences = normalized_text.count(keyword.casefold())
            if occurrences:
                total_matches += occurrences
                matched_keywords.append(f"{category}: {keyword}")

    score = min(100.0, (total_matches / AI_KEYWORD_SCORE_MATCHES_FOR_MAX) * 100.0)
    return round(score, 1), matched_keywords


@lru_cache(maxsize=1)
def get_text_classifier_bundle():
    if not TEXT_CLASSIFIER_MODEL:
        raise RuntimeError("Set TEXT_CLASSIFIER_MODEL to a fine-tuned binary text classifier model ID or local path.")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install text classification dependencies with: pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(TEXT_CLASSIFIER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(TEXT_CLASSIFIER_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def score_text_classifier(text):
    if not ENABLE_TEXT_CLASSIFIER or not TEXT_CLASSIFIER_MODEL:
        return None

    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return None

    try:
        import torch

        tokenizer, model, device = get_text_classifier_bundle()
        inputs = tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.shape[-1] == 1:
                probability = torch.sigmoid(logits).item()
            else:
                probabilities = torch.softmax(logits, dim=-1)
                probability = probabilities[0, 1].item() if probabilities.shape[-1] > 1 else probabilities.squeeze().item()

        return round(probability * 100.0, 1)
    except Exception as exc:
        print(f"Text classifier scoring failed: {exc}")
        return None


@lru_cache(maxsize=1)
def get_open_clip_model():
    try:
        import torch
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "Install image anomaly dependencies with: pip install open-clip-torch scikit-learn"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    model.to(device)
    return model, preprocess, device


@lru_cache(maxsize=1)
def get_dinov2_model():
    try:
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "Install image anomaly dependencies with: pip install torchvision scikit-learn"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, preprocess, device


def extract_image_embedding(image_bytes):
    if not image_bytes:
        return None

    try:
        import torch

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if IMAGE_EMBEDDING_BACKBONE == "dinov2":
            model, preprocess, device = get_dinov2_model()
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model(tensor)
        else:
            model, preprocess, device = get_open_clip_model()
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(tensor)

        vector = embedding.detach().cpu().numpy().astype("float32").reshape(-1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    except Exception as exc:
        print(f"Could not compute image embedding: {exc}")
        return None


def score_image_anomalies(image_uploads):
    results = {}

    for upload in image_uploads:
        participant_id = get_participant_id(upload.get("name"))
        if participant_id:
            results[participant_id] = {
                "score": None,
                "is_anomaly": False,
                "detector": IMAGE_ANOMALY_DETECTOR,
                "backbone": IMAGE_EMBEDDING_BACKBONE,
            }

    if not ENABLE_IMAGE_ANOMALY_SCORING:
        return results

    vectors = []
    participant_ids = []
    for upload in image_uploads:
        participant_id = get_participant_id(upload.get("name"))
        if not participant_id:
            continue
        vector = extract_image_embedding(upload.get("bytes"))
        if vector is None:
            continue
        vectors.append(vector)
        participant_ids.append(participant_id)

    if len(vectors) < 5:
        print("Image anomaly scoring skipped: need at least 5 valid image embeddings.")
        return results

    features = np.vstack(vectors)
    contamination = min(max(IMAGE_ANOMALY_CONTAMINATION, 0.001), 0.5)

    try:
        if IMAGE_ANOMALY_DETECTOR == "lof":
            from sklearn.neighbors import LocalOutlierFactor

            n_neighbors = max(2, min(20, len(features) - 1))
            detector = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            labels = detector.fit_predict(features)
            raw_scores = -detector.negative_outlier_factor_
        else:
            from sklearn.ensemble import IsolationForest

            detector = IsolationForest(
                n_estimators=300,
                contamination=contamination,
                random_state=42
            )
            labels = detector.fit_predict(features)
            raw_scores = -detector.score_samples(features)
    except Exception as exc:
        print(f"Image anomaly scoring failed: {exc}")
        return results

    raw_min = float(np.min(raw_scores))
    raw_max = float(np.max(raw_scores))
    if raw_max > raw_min:
        scaled_scores = ((raw_scores - raw_min) / (raw_max - raw_min)) * 100.0
    else:
        scaled_scores = np.zeros(len(raw_scores), dtype="float32")

    for index, participant_id in enumerate(participant_ids):
        score = float(scaled_scores[index])
        is_anomaly = labels[index] == -1 or score >= IMAGE_ANOMALY_FLAG_THRESHOLD
        results[participant_id] = {
            "score": round(score, 1),
            "is_anomaly": bool(is_anomaly),
            "detector": IMAGE_ANOMALY_DETECTOR,
            "backbone": IMAGE_EMBEDDING_BACKBONE
        }

    return results

# --- METADATA EXTRACTION ---
def analyze_image_metadata(image_bytes):
    if not image_bytes:
        image_bytes = b""

    metadata = {
        "MD5 Hash": hashlib.md5(image_bytes).hexdigest(),
        "Device/OS": "Unknown / Stripped Metadata",
        "Software Used": "None Detected",
        "Photo Created Time": None
    }
    if image_bytes == b"":
        return metadata

    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif_data = image._getexif()
        if exif_data:
            exif = {TAGS.get(key, key): val for key, val in exif_data.items()}
            
            make = exif.get("Make", "")
            model = exif.get("Model", "")
            if make or model:
                metadata["Device/OS"] = f"{make} {model}".strip()
                
            software = exif.get("Software", "")
            if software:
                metadata["Software Used"] = software
                
            photo_time = exif.get("DateTimeOriginal") or exif.get("DateTime")
            if photo_time:
                try:
                    metadata["Photo Created Time"] = datetime.strptime(photo_time, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
    except Exception:
        pass
    return metadata

# --- MAIN BATCH AUDIT ---
def run_security_audit(image_uploads):
    try:
        survey_df = pd.read_csv(SURVEY_CSV_PATH)
        survey_df["Start Time"] = pd.to_datetime(survey_df["Start Time"])
        survey_df["End Time"] = pd.to_datetime(survey_df["End Time"])
        survey_records_by_participant = survey_df.set_index("Participant ID").to_dict("index")
    except Exception as exc:
        print(f"Error loading {SURVEY_CSV_PATH}: {exc}")
        return

    report_rows = []
    technical_rows = []
    seen_image_hashes = {}
    image_anomaly_results = score_image_anomalies(image_uploads)

    for upload in image_uploads:
        filename = upload.get("name")
        image_bytes = upload.get("bytes")

        if not filename:
            print("Skipping upload with no filename.")
            continue

        participant_id = get_participant_id(filename)
        print(f"Auditing Participant: {participant_id}...")

        image_metadata = analyze_image_metadata(image_bytes)
        forgery_cues = compute_image_forgery_cues(image_bytes)
        image_hash = image_metadata["MD5 Hash"]
        security_flags = []

        duplicate_status = "No"
        is_duplicate_image = False
        if image_hash in seen_image_hashes:
            duplicate_status = f"YES (Matches {seen_image_hashes[image_hash]})"
            is_duplicate_image = True
            security_flags.append("Duplicate Image Fingerprint")
        else:
            seen_image_hashes[image_hash] = participant_id

        duration_minutes = None
        participant_missing_from_survey = False
        if participant_id in survey_records_by_participant:
            survey_record = survey_records_by_participant[participant_id]
            survey_start_time = survey_record["Start Time"]
            survey_end_time = survey_record["End Time"]
            duration_minutes = (survey_end_time - survey_start_time).total_seconds() / 60.0

            if duration_minutes < MINIMUM_MINUTES_REQUIRED:
                security_flags.append(
                    f"Speed Running ({duration_minutes:.1f} mins vs min {MINIMUM_MINUTES_REQUIRED})"
                )

            photo_created_time = image_metadata["Photo Created Time"]
            if isinstance(photo_created_time, datetime):
                if photo_created_time < survey_start_time:
                    security_flags.append("Anachronism (Photo taken before survey started)")
                elif photo_created_time > survey_end_time:
                    security_flags.append("Anachronism (Photo taken after survey submitted)")
        else:
            participant_missing_from_survey = True
            security_flags.append("ID mismatch (Participant not found in survey CSV export)")

        editing_software = image_metadata["Software Used"].lower()
        if any(app in editing_software for app in ["photoshop", "canva", "gimp", "illustrator"]):
            security_flags.append(f"Edited with software ({image_metadata['Software Used']})")
        metadata_missing = image_metadata["Device/OS"] == "Unknown / Stripped Metadata"

        latex_text = convert_math_photo_to_latex(
            image_bytes,
            filename=filename,
        )
        if not latex_text:
            security_flags.append("LaTeX extraction failed")

        extracted_text = latex_text.strip()
        ai_keyword_score, ai_keyword_matches = score_ai_keyword_flags(extracted_text)
        if ai_keyword_score >= 70.0:
            security_flags.append(f"High AI keyword signature ({ai_keyword_score}%)")

        text_classifier_score = score_text_classifier(extracted_text)
        if (
            text_classifier_score is not None
            and text_classifier_score >= TEXT_CLASSIFIER_THRESHOLD_PERCENT
        ):
            security_flags.append(f"High text classifier score ({text_classifier_score}%)")

        anomaly_result = image_anomaly_results.get(
            participant_id,
            {
                "score": None,
                "is_anomaly": False,
                "detector": IMAGE_ANOMALY_DETECTOR,
                "backbone": IMAGE_EMBEDDING_BACKBONE,
            },
        )
        if anomaly_result["is_anomaly"]:
            security_flags.append(
                "Image embedding anomaly "
                f"({anomaly_result['score']} via {anomaly_result['backbone']}/{anomaly_result['detector']})"
            )

        forgery_cue_score = forgery_cues["forgery_cue_score"]
        if forgery_cue_score is not None and forgery_cue_score >= 70.0:
            forgery_flag_reason = "High image forgery cue score"
            if forgery_cues["forgery_cue_flags"]:
                forgery_flag_reason += f" ({'; '.join(forgery_cues['forgery_cue_flags'])})"
            security_flags.append(forgery_flag_reason)

        metadata_corroborated = metadata_missing and (
            is_duplicate_image
            or anomaly_result["is_anomaly"]
            or (forgery_cue_score is not None and forgery_cue_score >= 50.0)
        )
        if metadata_corroborated:
            security_flags.append("Missing Device Metadata (corroborated by other risk cues)")

        duration_risk = compute_duration_risk(duration_minutes)
        heuristic_fusion_risk = compute_fusion_risk(
            ai_keyword_score=ai_keyword_score,
            text_classifier_score=text_classifier_score,
            image_anomaly_score=anomaly_result["score"],
            forgery_cue_score=forgery_cue_score,
            duration_risk=duration_risk,
            is_metadata_missing=metadata_missing,
            is_duplicate=is_duplicate_image,
            is_id_mismatch=participant_missing_from_survey,
        )

        trained_model_features = {
            "ai_keyword_score": ai_keyword_score,
            "text_classifier_score": text_classifier_score,
            "image_anomaly_score": anomaly_result["score"],
            "forgery_cue_score": forgery_cue_score,
            "duration_risk_score": duration_risk,
            "metadata_missing": 1.0 if metadata_missing else 0.0,
            "metadata_corroborated": 1.0 if metadata_corroborated else 0.0,
            "is_duplicate": 1.0 if is_duplicate_image else 0.0,
            "id_mismatch": 1.0 if participant_missing_from_survey else 0.0,
        }
        trained_model_fusion_risk = score_phase2_fusion_model(trained_model_features)
        if trained_model_fusion_risk is not None:
            fusion_risk = trained_model_fusion_risk
            fusion_source = "phase2_model"
        else:
            fusion_risk = heuristic_fusion_risk
            fusion_source = "phase1_heuristic"

        final_status = "FLAGGED FOR REVIEW" if security_flags else "APPROVED"
        security_flags_text = "; ".join(security_flags) if security_flags else "Clear"

        report_rows.append(
            build_review_row(
                participant_id=participant_id,
                security_flags=security_flags,
                duration_minutes=duration_minutes,
                participant_in_survey=not participant_missing_from_survey,
                duplicate_status=duplicate_status,
                photo_device=image_metadata["Device/OS"],
                metadata_missing=metadata_missing,
                math_read_successfully=bool(latex_text.strip()),
                work_preview=latex_text,
                ai_writing_score=ai_keyword_score,
                ai_writing_matches=ai_keyword_matches,
                text_ai_score=text_classifier_score,
                photo_integrity_score=forgery_cue_score,
                unusual_photo_score=anomaly_result["score"],
                overall_risk_score=fusion_risk["score"],
                risk_confidence=fusion_risk["confidence"],
                top_risk_drivers=fusion_risk["why_top"],
                priority=fusion_risk["tier"],
            )
        )

        technical_rows.append({
            "Participant ID": participant_id,
            "Duration (Mins)": format_report_value(
                round(duration_minutes, 1) if duration_minutes is not None else None
            ),
            "Device Detected": image_metadata["Device/OS"],
            "Is Duplicate": duplicate_status,
            "LaTeX Preview": latex_text[:150],
            "AI Keyword Score": ai_keyword_score,
            "AI Keyword Matches": "; ".join(ai_keyword_matches) if ai_keyword_matches else "None",
            "Text Classifier Score": format_report_value(text_classifier_score),
            "Text Classifier Model": TEXT_CLASSIFIER_MODEL or "Disabled",
            "Image Anomaly Score": format_report_value(anomaly_result["score"]),
            "Image Embedding Backbone": anomaly_result["backbone"],
            "Image Anomaly Detector": anomaly_result["detector"],
            "ELA Score": format_report_value(forgery_cues["ela_score"]),
            "JPEG Recompression Score": format_report_value(forgery_cues["jpeg_recompression_score"]),
            "Noise Residual Score": format_report_value(forgery_cues["noise_residual_score"]),
            "Blur Risk Score": format_report_value(forgery_cues["blur_risk_score"]),
            "Forgery Cue Score": format_report_value(forgery_cue_score),
            "Forgery Cue Flags": "; ".join(forgery_cues["forgery_cue_flags"]) if forgery_cues["forgery_cue_flags"] else "None",
            "Metadata Missing": "Yes" if metadata_missing else "No",
            "Metadata Corroborated": "Yes" if metadata_corroborated else "No",
            "Duration Risk Score": format_report_value(duration_risk),
            "Risk Fusion Score": format_report_value(fusion_risk["score"]),
            "Risk Fusion Confidence": format_report_value(fusion_risk["confidence"]),
            "Risk Fusion CI Low": fusion_risk.get("ci_low", "N/A"),
            "Risk Fusion CI High": fusion_risk.get("ci_high", "N/A"),
            "Risk Priority Tier": fusion_risk["tier"],
            "Risk Fusion Why": fusion_risk["why"],
            "Risk Fusion Top Contributors": fusion_risk["why_top"],
            "Risk Fusion Source": fusion_source,
            "Risk Fusion Model Version": fusion_risk.get("model_version", "phase1"),
            "Risk Fusion Model Trained At": fusion_risk.get("model_trained_at", "N/A"),
            "Risk Fusion Model Training Rows": fusion_risk.get("model_training_rows", "N/A"),
            "Security Flags": security_flags_text,
            "Final Status": final_status,
        })

    save_audit_reports(report_rows, technical_rows)


if __name__ == "__main__":
    try:
        print(f"Loading images from Dropbox folder: '{DROPBOX_SOURCE_FOLDER}'")
        image_uploads = load_dropbox_images()
        run_security_audit(image_uploads)
    except Exception as exc:
        print(f"Could not run audit: {exc}")
        print("Set DROPBOX_ACCESS_TOKEN and DROPBOX_SOURCE_FOLDER, then run again.")
