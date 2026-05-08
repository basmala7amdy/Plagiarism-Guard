from preprocessing.cleaner import clean_text
from nltk.tokenize import tokenize
from preprocessing.lemmatizer import lemmatize
from preprocessing.stopwords import remove_stopwords

def process(text):
    text = clean_text(text)
    sentences, words = tokenize(text)
    words = lemmatize(words)
    words = remove_stopwords(words)

    return {
        "text": text,
        "sentences": sentences,
        "tokens": words
    }