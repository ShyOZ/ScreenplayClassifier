# Imports
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from re import sub
from typing import List

# Methods
def build_corpus(text: str) -> str:
    porter_stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    corpus = sub("\'", "", text)                                        # Removes backslash-apostrophe
    corpus = sub("[^a-zA-Z]", " ", text)                                # Removes everything except alphabets

    words = [word for word in text.split() if word not in stop_words]   # Removes stopwords
    words = [porter_stemmer.stem(word) for word in words]               # Stems words
    corpus = "".join(words).lower()                                     # Lowercases the text

    return corpus
