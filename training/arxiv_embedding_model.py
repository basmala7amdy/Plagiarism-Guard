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
    return str(doc.get("original") or doc.get("text") or "").strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = [select_text(doc) for doc in docs]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, embeddings)

    print(f"Saved embeddings: {OUT_PATH}")


if __name__ == "__main__":
    main()