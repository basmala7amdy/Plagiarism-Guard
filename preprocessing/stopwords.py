from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))  # set for O(1) lookup

def remove_stopwords(words):
    return [w for w in words if w not in stop_words]