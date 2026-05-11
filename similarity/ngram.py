from nltk.util import ngrams


class NgramSimilarity:

    def __init__(self, n=3):
        self.n = n

    def get_ngrams(self, text):
        tokens = text.split()
        return set(ngrams(tokens, self.n))

    def compute(self, text1, text2):
        ngrams1 = self.get_ngrams(text1)
        ngrams2 = self.get_ngrams(text2)

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        if union == 0:
            return 0.0

        return intersection / union  # Jaccard similarity