from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSimilarity:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight sentence encoder

    def compute(self, text1, text2):
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])
        return float(cosine_similarity(emb1, emb2)[0][0])  # scalar cosine score