from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class CosineSimilarity:

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def compute(self, text1, text2):
        vectors = self.vectorizer.fit_transform([text1, text2])  # TF-IDF vectors
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]  # extract scalar
        return float(score)