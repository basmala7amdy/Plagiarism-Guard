import numpy as np
from nltk.tokenize import sent_tokenize


def best_matching_sentence(query, doc_text, sent_model):
    candidates = [s.strip() for s in sent_tokenize(doc_text) if s.strip()]

    if not candidates:
        return doc_text.strip()
    if len(candidates) == 1:
        return candidates[0]

    embs = sent_model.encode([query] + candidates, convert_to_numpy=True)
    q = embs[0]
    cand_embs = embs[1:]

    q_norm = np.linalg.norm(q) or 1.0
    c_norms = np.linalg.norm(cand_embs, axis=1)
    c_norms[c_norms == 0] = 1.0  # avoid divide-by-zero

    sims = (cand_embs @ q) / (c_norms * q_norm)  # cosine similarity scores
    return candidates[int(np.argmax(sims))]  # return best-matching sentence