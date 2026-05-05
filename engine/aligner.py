import difflib

class Aligner:

    def align(self, text1, text2):
        matcher = difflib.SequenceMatcher(None, text1, text2)

        matches = []
        for match in matcher.get_matching_blocks():
            matches.append({
                "a_start": match.a,
                "b_start": match.b,
                "size": match.size
            })

        return matches