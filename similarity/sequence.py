import difflib

class SequenceSimilarity:

    def compute(self, text1, text2):
        return difflib.SequenceMatcher(None, text1, text2).ratio()  # longest common subsequence ratio