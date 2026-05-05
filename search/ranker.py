class Ranker:
    def rank(self, results):
        return sorted(results, key=lambda x: x["score"], reverse=True)