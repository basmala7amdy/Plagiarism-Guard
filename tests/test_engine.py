from engine.detector import PlagiarismDetector


def test_plagiarism_detector():

    detector = PlagiarismDetector()

    query = "Deep learning is a subset of machine learning"

    docs = [
        "Machine learning includes deep learning",
        "I like football",
        "Deep learning is part of AI"
    ]

    result = detector.detect(query, docs)

    assert "final_score" in result
    assert result["final_score"] > 0.3


def test_best_match():

    detector = PlagiarismDetector()

    query = "Natural language processing is fun"

    docs = [
        "I play games",
        "Natural language processing is very fun",
        "Cooking recipes"
    ]

    result = detector.detect(query, docs)

    assert result["final_score"] > 0.5