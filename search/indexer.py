from sentence_transformers import SentenceTransformer

class Indexer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.doc_vectors = None
        self.documents = None

    def fit(self, documents):
        self.documents = documents
        texts = [doc["text"] for doc in documents]
        self.doc_vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=True
        )

    def transform(self, query):
        return self.model.encode([query], convert_to_numpy=True)