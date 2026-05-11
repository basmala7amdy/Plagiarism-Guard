import requests

BASE_URL = "http://127.0.0.1:5000"
TIMEOUT_SECONDS = 600  # BERT on CPU can take minutes for long docs


def check_plagiarism(text: str) -> dict:
    url = f"{BASE_URL}/check"
    payload = {"text": text}

    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to the backend.\n\n"
            "Make sure the FastAPI server is running:\n"
            "  uvicorn backend.main:app --host 127.0.0.1 --port 5000\n\n"
            "If it IS running, check that no HTTP_PROXY env var is "
            "redirecting localhost traffic (set $env:NO_PROXY = "
            "'127.0.0.1,localhost')."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"The backend did not respond within {TIMEOUT_SECONDS} seconds."
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if response.status_code == 400:
        try:
            detail = response.json().get("error", "Bad request")
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Backend validation error: {detail}")

    if not response.ok:
        raise RuntimeError(
            f"Backend returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Backend returned invalid JSON: {exc}") from exc


def upload_document(filename: str, data: bytes, mime: str | None = None) -> dict:
    url = f"{BASE_URL}/upload"
    files = {
        "file": (
            filename,
            data,
            mime or "application/octet-stream",
        )
    }

    try:
        response = requests.post(url, files=files, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to the backend.\n\n"
            "Make sure the FastAPI server is running:\n"
            "  uvicorn backend.main:app --host 127.0.0.1 --port 5000"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"The backend did not respond within {TIMEOUT_SECONDS} seconds."
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if not response.ok:
        try:
            detail = response.json().get("error", response.text[:200])
        except ValueError:
            detail = response.text[:200]
        raise RuntimeError(
            f"Backend returned HTTP {response.status_code}: {detail}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Backend returned invalid JSON: {exc}") from exc

    return body.get("result", body)  # unwrap nested result key if present


def ping_backend() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        return resp.ok  # True = backend is reachable
    except requests.exceptions.RequestException:
        return False
