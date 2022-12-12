# Imports
import re

from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Methods
def process_text(text: str):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    text = re.sub("\W+", " ", text)                                                           # Removes everything except alphabets
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Removes stopwords & stems

    return text