from .matcher import Matcher
from .scorer import Scorer
from training.infer import predict


class PlagiarismDetector:
    def __init__(self, threshold=0.6):
        self.matcher = Matcher()
        self.scorer = Scorer()
        self.threshold = threshold

    def detect(self, query, documents):
        pairs = self.matcher.match(query, documents)
        results = []

        for pair in pairs:
            hybrid_result = self.scorer.score(query, pair)
            model_result = predict(query, pair["doc_text"])

            search_score = float(pair.get("search_score", pair.get("score", 0.0)))
            hybrid_score = float(hybrid_result.get("final_score", 0.0))
            model_score = float(model_result.get("probabilities", {}).get("plagiarism", 0.0))

            final_score = (
                0.3 * search_score +
                0.3 * hybrid_score +
                0.4 * model_score
            )

            results.append({
                "doc_id": pair["doc_id"],
                "search_score": search_score,
                "hybrid_score": hybrid_score,
                "model_label": model_result.get("label"),
                "model_prediction": model_result.get("prediction"),
                "model_confidence": model_result.get("confidence"),
                "final_score": final_score,
                "text": pair["doc_text"]
            })

        if not results:
            return {
                "prediction": "not_plagiarism",
                "final_score": 0.0,
                "best_match": None,
                "details": []
            }

        best = max(results, key=lambda x: x["final_score"])

        return {
            "prediction": "plagiarism" if best["final_score"] >= self.threshold else "not_plagiarism",
            "final_score": best["final_score"],
            "best_match": best,
            "details": results
        }