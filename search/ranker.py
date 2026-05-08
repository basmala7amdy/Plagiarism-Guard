class Ranker:
    def rank(self, results):
        return sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)