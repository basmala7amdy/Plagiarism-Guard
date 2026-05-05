from similarity.hybrid_similarity import HybridSimilarity

class Scorer:

    def __init__(self):
        self.similarity = HybridSimilarity()

    def score(self, query, doc):
        return self.similarity.compute(query, doc)