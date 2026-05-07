from similarity.hybrid_similarity import HybridSimilarity

class Scorer:
    def __init__(self):
        self.similarity = HybridSimilarity()

    def score(self, query, doc):
        if isinstance(doc, dict):
            doc_text = doc.get("doc_text") or doc.get("full_text") or doc.get("text", "")
        else:
            doc_text = str(doc)

        result = self.similarity.compute(str(query), str(doc_text))

        if isinstance(result, dict):
            return {
                "cosine": float(result.get("cosine", result.get("cosine_score", 0.0))),
                "ngram": float(result.get("ngram", result.get("ngram_score", 0.0))),
                "semantic": float(result.get("semantic", result.get("semantic_score", 0.0))),
                "sequence": float(result.get("sequence", result.get("sequence_score", 0.0))),
                "final_score": float(result.get("final_score", result.get("score", 0.0)))
            }

        return {
            "final_score": float(result)
        }