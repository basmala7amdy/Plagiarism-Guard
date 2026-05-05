from .scorer import Scorer

class PlagiarismDetector:

    def __init__(self):
        self.scorer = Scorer()

    def detect(self, query, documents):

        results = []

        for doc in documents:
            score = self.scorer.score(query, doc)

            results.append({
                "doc": doc,
                "score": score["final_score"]
            })

        best = max(results, key=lambda x: x["score"])

        return {
            "final_score": best["score"],
            "details": results
        }