import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class SearchEngine:
    def __init__(self, indexer, documents):
        self.indexer = indexer
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]
        self.vectorizer = CountVectorizer(ngram_range=(2, 3))
        self.doc_ngram_matrix = self.vectorizer.fit_transform(self.texts)

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
        query_vec = self.indexer.transform(query)

        semantic_scores = cosine_similarity(
            query_vec, self.indexer.doc_vectors
        )[0]

        ngram_scores = self.compute_ngram_scores(query)

        semantic_scores = self.normalize(semantic_scores)
        ngram_scores = self.normalize(ngram_scores)

        final_scores = 0.7 * semantic_scores + 0.3 * ngram_scores

        top_k_idx = np.argsort(final_scores)[-top_k:][::-1]

        results = [
            {
                "doc_id": self.documents[i]["doc_id"],
                "text": self.documents[i]["text"][:200],
                "score": float(final_scores[i])
            }
            for i in top_k_idx
        ]

        return results