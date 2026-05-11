from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize(words):
    return [lemmatizer.lemmatize(w) for w in words]  # reduce words to base form