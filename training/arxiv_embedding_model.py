"""Build arxiv corpus embeddings from the *original* (raw) document text.

The earlier version of this script embedded the preprocessed `text` field
(lowercased + stopwords removed). That broke similarity scoring at query
time because user queries are raw natural language and never look like the
preprocessed corpus, so even a verbatim copy of an arxiv passage failed
to score as plagiarism.

This version embeds `doc["original"]` (with a fallback to `doc["text"]`
for any record where `original` is missing) so retrieval embeddings
match the distribution of real user inputs.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = REPO_ROOT / "data" / "procressed" / "arxiv_docs.json"
OUT_PATH = REPO_ROOT / "data" / "procressed" / "arxiv_embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def select_text(doc):
    original = str(doc.get("original", "") or "").strip()
    if original:
        return original
    return str(doc.get("text", "") or "").strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {MODEL_NAME!r} on device {device!r}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"Loading docs from {DOCS_PATH}")
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = [select_text(doc) for doc in docs]
    print(f"Encoding {len(texts):,} documents (batch_size={BATCH_SIZE})")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,  # consumers normalize as needed
    ).astype(np.float32)

    print(f"Saving embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, embeddings)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
