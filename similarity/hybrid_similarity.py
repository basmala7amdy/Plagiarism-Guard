from .cosine import CosineSimilarity
from .ngram import NgramSimilarity
from .sequence import SequenceSimilarity
from .semantic import SemanticSimilarity

class HybridSimilarity:

    def __init__(self):
        self.cosine = CosineSimilarity()
        self.ngram = NgramSimilarity()
        self.sequence = SequenceSimilarity()
        self.semantic = SemanticSimilarity()

    def compute(self, text1, text2):

        cos = self.cosine.compute(text1, text2)
        ngr = self.ngram.compute(text1, text2)
        seq = self.sequence.compute(text1, text2)
        sem = self.semantic.compute(text1, text2)

        final_score = (
            0.3 * cos +
            0.2 * ngr +
            0.2 * seq +
            0.3 * sem
        )

        return {
            "cosine": cos,
            "ngram": ngr,
            "sequence": seq,
            "semantic": sem,
            "final_score": round(final_score, 4)
        }