import os
import io
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
from PIL import Image, ImageOps, ImageChops, ImageFilter
from PIL.ExifTags import TAGS
from math_ocr_api import convert_math_photo_to_latex


# --- CONFIGURATION ---
MINIMUM_MINUTES_REQUIRED = 10.0  # Change to survey's minimum threshold
SURVEY_CSV_PATH = "survey_export.csv"
LOCAL_IMAGE_FOLDER = os.getenv("LOCAL_IMAGE_FOLDER", "test_uploads")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
ENABLE_IMAGE_ANOMALY_SCORING = os.getenv("ENABLE_IMAGE_ANOMALY_SCORING", "false").strip().lower() in {"1", "true", "yes", "on"}
IMAGE_EMBEDDING_BACKBONE = os.getenv("IMAGE_EMBEDDING_BACKBONE", "open_clip").strip().lower()
IMAGE_ANOMALY_DETECTOR = os.getenv("IMAGE_ANOMALY_DETECTOR", "isolation_forest").strip().lower()
IMAGE_ANOMALY_CONTAMINATION = float(os.getenv("IMAGE_ANOMALY_CONTAMINATION", "0.1"))
IMAGE_ANOMALY_FLAG_THRESHOLD = float(os.getenv("IMAGE_ANOMALY_FLAG_THRESHOLD", "70.0"))
TEXT_CLASSIFIER_MODEL = os.getenv("TEXT_CLASSIFIER_MODEL")
ENABLE_TEXT_CLASSIFIER = os.getenv("ENABLE_TEXT_CLASSIFIER", "false").strip().lower() in {"1", "true", "yes", "on"} or bool(TEXT_CLASSIFIER_MODEL)
TEXT_CLASSIFIER_THRESHOLD = float(os.getenv("TEXT_CLASSIFIER_THRESHOLD", "0.7"))
FUSION_WEIGHT_TEXT = float(os.getenv("FUSION_WEIGHT_TEXT", "0.30"))
FUSION_WEIGHT_FORGERY = float(os.getenv("FUSION_WEIGHT_FORGERY", "0.25"))
FUSION_WEIGHT_ANOMALY = float(os.getenv("FUSION_WEIGHT_ANOMALY", "0.20"))
FUSION_WEIGHT_DURATION = float(os.getenv("FUSION_WEIGHT_DURATION", "0.10"))
FUSION_WEIGHT_METADATA = float(os.getenv("FUSION_WEIGHT_METADATA", "0.10"))
FUSION_WEIGHT_DUPLICATE = float(os.getenv("FUSION_WEIGHT_DUPLICATE", "0.05"))

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

# --- GOOGLE DRIVE ACCESS ---
def get_google_drive_service():
    if not GOOGLE_SERVICE_ACCOUNT_FILE:
        raise ValueError("Set GOOGLE_SERVICE_ACCOUNT_FILE to your service account JSON file path.")

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError("Install Google Drive dependencies with: pip install -r requirements.txt") from exc

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = service_account.Credentials.from_service_account_file(
        GOOGLE_SERVICE_ACCOUNT_FILE,
        scopes=scopes
    )
    return build("drive", "v3", credentials=credentials)


def load_google_drive_images():
    if not GOOGLE_DRIVE_FOLDER_ID:
        raise ValueError("Set GOOGLE_DRIVE_FOLDER_ID to the Drive folder that receives uploaded photos.")

    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise ImportError("Install Google Drive dependencies with: pip install -r requirements.txt") from exc

    service = get_google_drive_service()
    query = (
        f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents "
        "and trashed = false "
        "and mimeType contains 'image/'"
    )
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=1000
    ).execute()

    drive_items = []
    for file_info in results.get("files", []):
        request = service.files().get_media(fileId=file_info["id"])
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        drive_items.append({
            "name": file_info["name"],
            "bytes": buffer.getvalue()
        })
    return drive_items


def load_local_images(folder_path=LOCAL_IMAGE_FOLDER):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Local image folder not found: {folder_path}")

    supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    local_items = []
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        _, extension = os.path.splitext(filename)
        if extension.lower() not in supported_extensions:
            continue

        with open(file_path, "rb") as image_file:
            local_items.append({
                "name": filename,
                "bytes": image_file.read(),
                "path": file_path
            })

    if not local_items:
        raise ValueError(f"No image files found in local folder: {folder_path}")

    return local_items


# --- OCR AND AI ANALYSIS ---


def load_latex_sidecar(image_path):
    if not image_path:
        return ""

    base_path, _ = os.path.splitext(image_path)
    sidecar_path = f"{base_path}.latex.txt"
    if not os.path.isfile(sidecar_path):
        return ""

    try:
        with open(sidecar_path, "r", encoding="utf-8") as sidecar_file:
            return sidecar_file.read().strip()
    except Exception as exc:
        print(f"Could not read LaTeX sidecar {sidecar_path}: {exc}")
        return ""


def clamp_score(value):
    return max(0.0, min(100.0, float(value)))


def compute_image_forgery_cues(image_bytes):
    default = {
        "ela_score": None,
        "jpeg_recompression_score": None,
        "noise_residual_score": None,
        "blur_risk_score": None,
        "forgery_cue_score": None,
        "forgery_cue_flags": [],
    }
    if not image_bytes:
        return default

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

        q95_buffer = io.BytesIO()
        q75_buffer = io.BytesIO()
        image.save(q95_buffer, format="JPEG", quality=95)
        image.save(q75_buffer, format="JPEG", quality=75)
        q95 = np.asarray(Image.open(io.BytesIO(q95_buffer.getvalue())).convert("RGB"), dtype=np.float32)
        q75 = np.asarray(Image.open(io.BytesIO(q75_buffer.getvalue())).convert("RGB"), dtype=np.float32)
        jpeg_gap = float(np.mean(np.abs(q95 - q75)))
        jpeg_recompression_score = clamp_score(jpeg_gap * 2.5)

        gray = image.convert("L")
        gray_np = np.asarray(gray, dtype=np.float32)
        blurred_np = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32)
        residual = gray_np - blurred_np
        residual_std = float(np.std(residual))
        noise_residual_score = clamp_score(residual_std * 4.0)

        gy, gx = np.gradient(gray_np)
        gradient_mag = np.sqrt((gx * gx) + (gy * gy))
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
        return default


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
    if score >= 75:
        tier = "High"
    elif score >= 50:
        tier = "Medium"
    else:
        tier = "Low"

    sorted_components = sorted(weighted_components, key=lambda item: item["weighted"], reverse=True)
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


def score_image_anomalies(items):
    default_detector = IMAGE_ANOMALY_DETECTOR
    default_backbone = IMAGE_EMBEDDING_BACKBONE
    results = {}

    for item in items:
        participant_id = get_participant_id(item.get("name"))
        if participant_id:
            results[participant_id] = {
                "score": None,
                "is_anomaly": False,
                "detector": default_detector,
                "backbone": default_backbone
            }

    if not ENABLE_IMAGE_ANOMALY_SCORING:
        return results

    vectors = []
    participant_ids = []
    for item in items:
        participant_id = get_participant_id(item.get("name"))
        if not participant_id:
            continue
        vector = extract_image_embedding(item.get("bytes"))
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

# --- MAIN BATCH EXECUTIVE ---
def run_security_audit(google_drive_items):
    # Load and parse the survey platform times
    try:
        survey_df = pd.read_csv(SURVEY_CSV_PATH)
        # Automatically handle standard datetime formats
        survey_df['Start Time'] = pd.to_datetime(survey_df['Start Time'])
        survey_df['End Time'] = pd.to_datetime(survey_df['End Time'])
        # Set Participant ID as index for fast lookups
        survey_lookup = survey_df.set_index('Participant ID').to_dict('index')
    except Exception as e:
        print(f"Error loading {SURVEY_CSV_PATH}: {e}")
        return

    report_data = []
    seen_hashes = {}
    image_anomaly_results = score_image_anomalies(google_drive_items)

    for item in google_drive_items:
        # Assuming filename structure matches the ID format in your survey CSV
        filename = item.get('name')
        image_bytes = item.get('bytes') # Placeholder for downloaded file data from Drive API

        if not filename:
            print("Skipping Drive item with no filename.")
            continue

        participant_id = get_participant_id(filename)
        
        print(f"Auditing Participant: {participant_id}...")
        
        # 1. Image Analytics
        meta = analyze_image_metadata(image_bytes)
        forgery_cues = compute_image_forgery_cues(image_bytes)
        current_hash = meta["MD5 Hash"]
        flag_reasons = []
        
        # 2. Check for Duplicate Uploads
        is_duplicate = "No"
        is_duplicate_flag = False
        if current_hash in seen_hashes:
            is_duplicate = f"YES (Matches {seen_hashes[current_hash]})"
            is_duplicate_flag = True
            flag_reasons.append("Duplicate Image Fingerprint")
        else:
            seen_hashes[current_hash] = participant_id

        # 3. Pull Survey Platform Durations
        duration_minutes = None
        id_mismatch = False
        if participant_id in survey_lookup:
            timestamps = survey_lookup[participant_id]
            start_t = timestamps['Start Time']
            end_t = timestamps['End Time']
            
            # Calculate overall survey duration
            duration_minutes = (end_t - start_t).total_seconds() / 60.0
            
            if duration_minutes < MINIMUM_MINUTES_REQUIRED:
                flag_reasons.append(f"Speed Running ({duration_minutes:.1f} mins vs min {MINIMUM_MINUTES_REQUIRED})")
                
            # 4. Image Creation Time vs. Survey Window Verification
            img_time = meta["Photo Created Time"]
            if isinstance(img_time, datetime):
                if img_time < start_t:
                    flag_reasons.append("Anachronism (Photo taken before survey started)")
                elif img_time > end_t:
                    flag_reasons.append("Anachronism (Photo taken after survey submitted)")
        else:
            id_mismatch = True
            flag_reasons.append("ID mismatch (Participant not found in survey CSV export)")

        # 5. Metadata Software Flags
        software_lower = meta["Software Used"].lower()
        if any(app in software_lower for app in ["photoshop", "canva", "gimp", "illustrator"]):
            flag_reasons.append(f"Edited with software ({meta['Software Used']})")
        metadata_missing = meta["Device/OS"] == "Unknown / Stripped Metadata"

        # 6. OCR, Math-to-LaTeX, and AI-style keyword checks
        latex_text = load_latex_sidecar(item.get("path")) or convert_math_photo_to_latex(
            image_bytes,
            filename=filename,
        )
        if not latex_text:
            flag_reasons.append("LaTeX extraction failed")

        ocr_text = ""
        combined_text = latex_text.strip()
        ai_keyword_score, ai_keyword_matches = score_ai_keyword_flags(combined_text)
        if ai_keyword_score >= 70.0:
            flag_reasons.append(f"High AI keyword signature ({ai_keyword_score}%)")

        text_classifier_score = score_text_classifier(combined_text)
        if text_classifier_score is not None and text_classifier_score >= (TEXT_CLASSIFIER_THRESHOLD * 100.0):
            flag_reasons.append(
                f"High text classifier score ({text_classifier_score}%)"
            )

        anomaly_result = image_anomaly_results.get(participant_id, {
            "score": None,
            "is_anomaly": False,
            "detector": IMAGE_ANOMALY_DETECTOR,
            "backbone": IMAGE_EMBEDDING_BACKBONE
        })
        if anomaly_result["is_anomaly"]:
            flag_reasons.append(
                "Image embedding anomaly "
                f"({anomaly_result['score']} via {anomaly_result['backbone']}/{anomaly_result['detector']})"
            )

        if forgery_cues["forgery_cue_score"] is not None and forgery_cues["forgery_cue_score"] >= 70.0:
            reason = "High image forgery cue score"
            if forgery_cues["forgery_cue_flags"]:
                reason += f" ({'; '.join(forgery_cues['forgery_cue_flags'])})"
            flag_reasons.append(reason)

        # Missing metadata is a soft signal; only escalate when corroborated.
        metadata_corroborated = metadata_missing and (
            is_duplicate_flag
            or anomaly_result["is_anomaly"]
            or (
                forgery_cues["forgery_cue_score"] is not None
                and forgery_cues["forgery_cue_score"] >= 50.0
            )
        )
        if metadata_corroborated:
            flag_reasons.append("Missing Device Metadata (corroborated by other risk cues)")

        duration_risk = compute_duration_risk(duration_minutes)
        fusion_risk = compute_fusion_risk(
            ai_keyword_score=ai_keyword_score,
            text_classifier_score=text_classifier_score,
            image_anomaly_score=anomaly_result["score"],
            forgery_cue_score=forgery_cues["forgery_cue_score"],
            duration_risk=duration_risk,
            is_metadata_missing=metadata_missing,
            is_duplicate=is_duplicate_flag,
            is_id_mismatch=id_mismatch,
        )

        # Determine Final Review State
        status = "FLAGGED FOR REVIEW" if flag_reasons else "APPROVED"

        report_data.append({
            "Participant ID": participant_id,
            "Duration (Mins)": round(duration_minutes, 1) if duration_minutes is not None else "N/A",
            "Device Detected": meta["Device/OS"],
            "Is Duplicate": is_duplicate,
            "LaTeX Preview": latex_text[:150],
            "AI Keyword Score": ai_keyword_score,
            "AI Keyword Matches": "; ".join(ai_keyword_matches) if ai_keyword_matches else "None",
            "Text Classifier Score": text_classifier_score if text_classifier_score is not None else "N/A",
            "Text Classifier Model": TEXT_CLASSIFIER_MODEL or "Disabled",
            "Image Anomaly Score": anomaly_result["score"] if anomaly_result["score"] is not None else "N/A",
            "Image Embedding Backbone": anomaly_result["backbone"],
            "Image Anomaly Detector": anomaly_result["detector"],
            "ELA Score": forgery_cues["ela_score"] if forgery_cues["ela_score"] is not None else "N/A",
            "JPEG Recompression Score": forgery_cues["jpeg_recompression_score"] if forgery_cues["jpeg_recompression_score"] is not None else "N/A",
            "Noise Residual Score": forgery_cues["noise_residual_score"] if forgery_cues["noise_residual_score"] is not None else "N/A",
            "Blur Risk Score": forgery_cues["blur_risk_score"] if forgery_cues["blur_risk_score"] is not None else "N/A",
            "Forgery Cue Score": forgery_cues["forgery_cue_score"] if forgery_cues["forgery_cue_score"] is not None else "N/A",
            "Forgery Cue Flags": "; ".join(forgery_cues["forgery_cue_flags"]) if forgery_cues["forgery_cue_flags"] else "None",
            "Metadata Missing": "Yes" if metadata_missing else "No",
            "Metadata Corroborated": "Yes" if metadata_corroborated else "No",
            "Duration Risk Score": duration_risk if duration_risk is not None else "N/A",
            "Risk Fusion Score": fusion_risk["score"] if fusion_risk["score"] is not None else "N/A",
            "Risk Fusion Confidence": fusion_risk["confidence"] if fusion_risk["confidence"] is not None else "N/A",
            "Risk Priority Tier": fusion_risk["tier"],
            "Risk Fusion Why": fusion_risk["why"],
            "Risk Fusion Top Contributors": fusion_risk["why_top"],
            "Security Flags": "; ".join(flag_reasons) if flag_reasons else "Clear",
            "Final Status": status
        })

    # Save finalized document
    df = pd.DataFrame(report_data)
    df.to_csv("comprehensive_fraud_report.csv", index=False)
    print("Audit completely finished. Open 'comprehensive_fraud_report.csv' to review flags.")


if __name__ == "__main__":
    try:
        if GOOGLE_DRIVE_FOLDER_ID and GOOGLE_SERVICE_ACCOUNT_FILE:
            audit_items = load_google_drive_images()
        else:
            print(f"Google Drive is not configured. Loading images from '{LOCAL_IMAGE_FOLDER}' instead.")
            audit_items = load_local_images()
        run_security_audit(audit_items)
    except Exception as exc:
        print(f"Could not run audit: {exc}")
        print("Set Google Drive variables or place test images in LOCAL_IMAGE_FOLDER, then run again.")
