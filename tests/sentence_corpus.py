import json
import sys
from pathlib import Path
from engine.aligner import best_matching_sentence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from engine.detector import PlagiarismDetector


def load_arxiv_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            preprocessed = str(item.get("text", "")).strip()
            original = str(item.get("original", "")).strip() or preprocessed
            doc_id = item.get("doc_id", f"arxiv_{i}")
        else:
            preprocessed = str(item).strip()
            original = preprocessed
            doc_id = f"arxiv_{i}"

        if preprocessed:
            documents.append({"doc_id": doc_id, "preprocessed_text": preprocessed, "original": original})

    return documents


class NpySearchEngine:
    def __init__(self, documents, embeddings_path, model_name="all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = np.load(embeddings_path).astype(np.float32)

        if len(self.documents) != len(self.embeddings):
            raise ValueError(
                f"Documents count ({len(self.documents)}) does not match embeddings count ({len(self.embeddings)}). "
                "Make sure embeddings were saved in the same order as the documents."
            )

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero
        self.embeddings = self.embeddings / norms  # normalise for cosine similarity

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)[0]

        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            q_norm = 1.0
        query_vec = query_vec / q_norm

        scores = np.dot(self.embeddings, query_vec)  # cosine similarity via dot product
        top_indices = np.argsort(scores)[::-1][:top_k]  # top-k indices by score

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            original_text = doc.get("original") or doc.get("preprocessed_text", "")
            results.append({
                "doc_id": doc["doc_id"],
                "text": original_text,
                "full_text": original_text,
                "preprocessed_text": doc.get("preprocessed_text", ""),
                "search_score": float(scores[idx]),
            })

        return results


def analyze_paragraph(paragraph, search_engine, detector, top_k=5):
    sentences = sent_tokenize(paragraph)
    results = []

    for i, sentence in enumerate(sentences, start=1):
        sentence = sentence.strip()
        if not sentence:
            continue

        top_docs = search_engine.search(sentence, top_k=top_k)

        for doc in top_docs:
            best_sent = best_matching_sentence(sentence, doc.get("full_text", ""), search_engine.model)
            doc["text"] = best_sent
            doc["full_text"] = best_sent

        detection = detector.detect(sentence, top_docs)
        best_match = detection.get("best_match")

        results.append({
            "sentence_id": i,
            "sentence": sentence,
            "prediction": detection["prediction"],
            "sentence_plagiarism_percentage": round(detection["final_score"] * 100, 2),
            "best_match": best_match,
            "details": detection["details"]
        })

    return results


def print_summary(results, sentence_preview=80):
    if not results:
        print("\nNo sentences analysed.")
        return

    plagiarised = [r for r in results if r["prediction"] == "plagiarism"]
    avg_pct = sum(r["sentence_plagiarism_percentage"] for r in results) / len(results)

    bar = "=" * 78
    print(f"\n{bar}\nSUMMARY\n{bar}")
    print(
        f"sentences analysed : {len(results)}\n"
        f"flagged plagiarism : {len(plagiarised)} / {len(results)}\n"
        f"avg score          : {avg_pct:.2f}%"
    )
    print("-" * 78)
    print(f"{'#':>2}  {'verdict':<15}  {'score':>6}  sentence")
    print("-" * 78)
    for r in results:
        sent = r["sentence"].replace("\n", " ").strip()
        if len(sent) > sentence_preview:
            sent = sent[: sentence_preview - 3] + "..."
        print(f"{r['sentence_id']:>2}  {r['prediction']:<15}  {r['sentence_plagiarism_percentage']:>5.1f}%  {sent}")
    print(bar)


if __name__ == "__main__":
    docs_path = r"C:\Users\AmrAhmed\Documents\GitHub\Plagiarism-Guard\data\procressed\arxiv_docs.json"
    emb_path = r"C:\Users\AmrAhmed\Documents\GitHub\Plagiarism-Guard\data\procressed\arxiv_embeddings.npy"

    documents = load_arxiv_documents(docs_path)
    search_engine = NpySearchEngine(documents=documents, embeddings_path=emb_path, model_name="all-MiniLM-L6-v2")
    detector = PlagiarismDetector()

    paragraph = ("The on-line shortest path problem is considered under various models of\npartial monitoring. Given a weighted directed acyclic graph whose edge weights\ncan change in an arbitrary (adversarial) way, a decision maker has to choose in\neach round of a game a path between two distinguished vertices such that the\nloss of the chosen path (defined as the sum of the weights of its composing\nedges) be as small as possible.")

    result = analyze_paragraph(paragraph, search_engine, detector, top_k=5)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print_summary(result)