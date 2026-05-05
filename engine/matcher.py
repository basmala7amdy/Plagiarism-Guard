class Matcher:

    def match(self, query, documents):
        pairs = []
        for doc in documents:
            pairs.append((query, doc))
        return pairs