"""SimpleTex-only math OCR helper."""

import json
import os
import ssl
import urllib.error
import urllib.request

try:
    import certifi

    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

_ENV_LOADED = False


def load_project_env():
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    _ENV_LOADED = True
    search_roots = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for root in search_roots:
        env_path = os.path.join(root, ".env")
        if not os.path.isfile(env_path):
            continue

        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
        break


def get_simpletex_settings():
    load_project_env()
    return {
        "uat": os.getenv("SIMPLETEX_UAT", "").strip(),
        "api_url": os.getenv("SIMPLETEX_API_URL", "https://server.simpletex.cn/api/simpletex_ocr"),
        "rec_mode": os.getenv("SIMPLETEX_REC_MODE", "formula").strip().lower(),
        "enable_img_rot": os.getenv("SIMPLETEX_ENABLE_IMG_ROT", "true").strip().lower(),
    }


def convert_math_photo_to_latex(image_bytes, filename="upload.jpg"):
    settings = get_simpletex_settings()
    if not image_bytes or not settings["uat"]:
        return ""

    content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    boundary = f"----SimpleTexBoundary{os.urandom(12).hex()}"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="rec_mode"\r\n\r\n')
    body.extend(f"{settings['rec_mode']}\r\n".encode())
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="enable_img_rot"\r\n\r\n')
    body.extend(f"{settings['enable_img_rot']}\r\n".encode())
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode())
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
    body.extend(image_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        settings["api_url"],
        data=bytes(body),
        headers={
            "token": settings["uat"],
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90, context=SSL_CONTEXT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"SimpleTex OCR failed ({exc.code}): {error_body}")
        return ""
    except Exception as exc:
        print(f"SimpleTex OCR failed: {exc}")
        return ""

    if not payload.get("status"):
        return ""
    response_data = payload.get("res", {})
    if isinstance(response_data, dict):
        info = str(response_data.get("info", "")).strip()
        if info:
            return info
        latex = str(response_data.get("latex", "")).strip()
        if latex:
            return latex
        return str(response_data.get("text", "")).strip()
    return ""
