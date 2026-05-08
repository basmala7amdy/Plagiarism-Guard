"""
PlagiarismGuard - API Client
===========================
Handles ALL communication between the Streamlit frontend and
the Flask backend. No other file should import 'requests' directly.

Data flow:
  ui.py  →  api_client.check_plagiarism(text)
          →  POST http://127.0.0.1:5000/check  { "text": "..." }
          ←  JSON response
          →  returns Python dict  (or raises RuntimeError on failure)
"""

import requests

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:5000"
TIMEOUT_SECONDS = 30          # give the backend plenty of time


# ── Public API ────────────────────────────────────────────────────────────────

def check_plagiarism(text: str) -> dict:
    """
    Send *text* to the Flask /check endpoint and return the parsed response.

    Parameters
    ----------
    text : str
        The user-supplied text to analyse.

    Returns
    -------
    dict
        Parsed JSON response from the backend, e.g.::

            {
                "score": 87,
                "sources": [{"title": "...", "url": "...", "score": 0.91}, ...],
                "highlighted_text": "<mark>...</mark>",
                "chart_data": {
                    "matched": 87,
                    "original": 13,
                    "source_breakdown": [...],
                    "similarity_timeline": [...]
                },
                "word_count": 120,
                "sentence_count": 8,
                "unique_phrases": 14
            }

    Raises
    ------
    RuntimeError
        On any network error, non-200 status, or malformed JSON.
    """
    url = f"{BASE_URL}/check"
    payload = {"text": text}

    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "❌ Cannot connect to the backend.\n\n"
            "Make sure the Flask server is running:\n"
            "  cd backend && python app.py"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"⏱️ The backend did not respond within {TIMEOUT_SECONDS} seconds."
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    # Surface backend validation errors cleanly
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


def ping_backend() -> bool:
    """
    Return True if the backend health-check endpoint is reachable.
    Used by ui.py to show a status indicator without blocking the UI.
    """
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=3)
        return resp.ok
    except requests.exceptions.RequestException:
        return False
