# Imports
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from re import sub
from typing import List

# Methods
def process_text(text: str) -> str:
    # TODO: COMPLETE
    stop_words = set(stopwords.words("english"))
    # porter_stemmer = PorterStemmer()

    text = sub("[^a-zA-Z]", " ", text)                                          # Removes everything except alphabets
    text = " ".join([word for word in text.split() if word not in stop_words])  # Removes stopwords + spaces
    text = sub("\'", "", text)                                                  # Removes backslash-apostrophe

    return text.lower()
