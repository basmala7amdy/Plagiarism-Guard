from similarity.cosine import CosineSimilarity
from similarity.ngram import NgramSimilarity
from similarity.sequence import SequenceSimilarity
from similarity.semantic import SemanticSimilarity
from similarity.hybrid_similarity import HybridSimilarity


text1 = "Machine learning is amazing"
text2 = "Machine learning is awesome"
text3 = "I love pizza"


def test_cosine_similarity():
    model = CosineSimilarity()
    score = model.compute(text1, text2)

    assert 0 <= score <= 1
    assert score > 0.3  


def test_ngram_similarity():
    model = NgramSimilarity()
    score = model.compute(text1, text2)

    assert 0 <= score <= 1


def test_sequence_similarity():
    model = SequenceSimilarity()
    score = model.compute(text1, text2)

    assert score > 0.3


def test_semantic_similarity():
    model = SemanticSimilarity()
    score = model.compute(text1, text2)

    assert score > 0.5   


def test_hybrid_similarity():
    model = HybridSimilarity()
    result = model.compute(text1, text2)

    assert "final_score" in result
    assert result["final_score"] > 0.4


def test_different_texts():
    model = HybridSimilarity()
    result = model.compute(text1, text3)

    assert result["final_score"] < 0.5