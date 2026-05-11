import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from .ranker import Ranker


class SearchEngine:
    def __init__(self, indexer, documents):
        self.indexer = indexer
        self.documents = documents

        self.texts = [
            str(doc.get("original") or doc.get("text", ""))
            for doc in documents
        ]

        self.vectorizer = CountVectorizer(ngram_range=(2, 3))
        self.doc_ngram_matrix = self.vectorizer.fit_transform(self.texts)  # precompute n-gram matrix
        self.ranker = Ranker()

    def normalize(self, scores):
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val == 0:
            return scores
        return (scores - min_val) / (max_val - min_val)

    def compute_ngram_scores(self, query):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_ngram_matrix)[0]
        return scores

    def search(self, query, top_k=5):
        query = str(query).strip()
        query_vec = self.indexer.transform(query)

        semantic_scores = cosine_similarity(query_vec, self.indexer.doc_vectors)[0]
        ngram_scores = self.compute_ngram_scores(query)

        semantic_scores = self.normalize(semantic_scores)
        ngram_scores = self.normalize(ngram_scores)

        final_scores = 0.7 * semantic_scores + 0.3 * ngram_scores  # weighted blend

        results = []
        for i, doc in enumerate(self.documents):
            original_text = str(doc.get("original") or doc.get("text", ""))
            results.append({
                "doc_id": doc.get("doc_id", f"doc_{i}"),
                "document": doc,
                "text": original_text[:200],
                "full_text": original_text,
                "semantic_score": float(semantic_scores[i]),
                "ngram_score": float(ngram_scores[i]),
                "score": float(final_scores[i]),
                "search_score": float(final_scores[i]),
            })

        results = self.ranker.rank(results)
        return results[:top_k]