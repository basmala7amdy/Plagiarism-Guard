from sentence_transformers import SentenceTransformer

class Indexer:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_vectors = None
        self.documents = None

    def fit(self, documents):
        self.documents = documents
        texts = [str(doc.get("text", "")) for doc in documents]
        self.doc_vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=True
        )
        return self

    def transform(self, query):
        return self.model.encode([str(query)], convert_to_numpy=True)