class Matcher:
    def match(self, query, documents):
        pairs = []

        for doc in documents:
            if isinstance(doc, dict):
                doc_text = doc.get("full_text") or doc.get("text", "")
                doc_id = doc.get("doc_id")
                raw_score = doc.get("search_score", doc.get("score", 0.0))
                search_score = float(raw_score) if raw_score is not None else 0.0
            else:
                doc_text = str(doc)
                doc_id = None
                search_score = 0.0

            pairs.append({
                "query": str(query),
                "doc_id": doc_id,
                "doc_text": str(doc_text),
                "search_score": search_score,
                "raw_doc": doc
            })

        return pairs