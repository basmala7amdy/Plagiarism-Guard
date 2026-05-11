import argparse
import json
import sys
from pathlib import Path

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.aligner import best_matching_sentence
from engine.detector import PlagiarismDetector


class NpySearchEngine:
    def __init__(self, documents, embeddings_path, model_name="all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = np.load(embeddings_path).astype(np.float32)

        if len(self.documents) != len(self.embeddings):
            raise ValueError(
                f"Documents count ({len(self.documents)}) does not match embeddings count ({len(self.embeddings)})."
            )

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero
        self.embeddings = self.embeddings / norms  # normalise for cosine similarity

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)[0]

        q_norm = np.linalg.norm(query_vec) or 1.0
        query_vec = query_vec / q_norm

        scores = np.dot(self.embeddings, query_vec)  # cosine similarity via dot product
        top_indices = np.argsort(scores)[::-1][:top_k]  # indices of top-k scores

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            original_text = str(doc.get("original") or doc.get("text", ""))
            results.append({
                "doc_id": doc.get("doc_id", f"doc_{idx}"),
                "text": original_text,
                "full_text": original_text,
                "preprocessed_text": doc.get("preprocessed_text", doc.get("text", "")),
                "search_score": float(scores[idx]),
            })

        return results


def load_arxiv_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            preprocessed = str(item.get("text", "") or "").strip()
            original = str(item.get("original", "") or "").strip() or preprocessed
            doc_id = item.get("doc_id", f"arxiv_{i}")
        else:
            preprocessed = str(item).strip()
            original = preprocessed
            doc_id = f"arxiv_{i}"

        if preprocessed or original:
            documents.append({"doc_id": doc_id, "preprocessed_text": preprocessed, "original": original})

    return documents


def analyze_text(text, search_engine, detector, top_k=5):
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    results = []

    for i, sentence in enumerate(sentences, start=1):
        top_docs = search_engine.search(sentence, top_k=top_k)

        refined_docs = []
        for doc in top_docs:
            best_sent = best_matching_sentence(sentence, doc.get("full_text", ""), search_engine.model)
            refined_docs.append({**doc, "candidate_text": best_sent, "text": best_sent, "full_text": best_sent})

        detection = detector.detect(sentence, refined_docs)

        results.append({
            "sentence_id": i,
            "sentence": sentence,
            "prediction": detection["prediction"],
            "sentence_plagiarism_percentage": round(detection["final_score"] * 100, 2),
            "best_match": detection.get("best_match"),
            "details": detection.get("details", []),
        })

    plagiarised = sum(1 for r in results if r["prediction"] == "plagiarism")
    avg_score = (  # average plagiarism score across all sentences
        sum(r["sentence_plagiarism_percentage"] for r in results) / len(results)
        if results else 0.0
    )

    return {
        "sentences_analyzed": len(results),
        "flagged_plagiarism": plagiarised,
        "average_score": round(avg_score, 2),
        "results": results,
    }


DOCS_PATH = REPO_ROOT / "data" / "procressed" / "arxiv_docs.json"
EMB_PATH = REPO_ROOT / "data" / "procressed" / "arxiv_embeddings.npy"

_documents = load_arxiv_documents(DOCS_PATH)
_search_engine = NpySearchEngine(_documents, EMB_PATH)
_detector = PlagiarismDetector()


def predict_plagiarism(text, top_k=5):
    text = str(text).strip()
    if not text:
        return {"error": "No text provided"}
    return analyze_text(text=text, search_engine=_search_engine, detector=_detector, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Run plagiarism detection on plain text input.")
    parser.add_argument("text", nargs="?", default=None, help="Raw text to analyze.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest documents to inspect")
    args = parser.parse_args()

    text = args.text
    if text is None:
        print("Enter text:")
        text = input()

    output = predict_plagiarism(text, top_k=args.top_k)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()