# Imports
import nltk, re

from nltk.corpus import stopwords
from typing import Dict, Set, Tuple
from collections import defaultdict
# from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

# Methods
def process_text(text: str) -> str:
    # TODO: COMPLETE
    stop_words = set(stopwords.words("english"))
    # porter_stemmer = PorterStemmer()

    text = re.sub("\W+", " ", text)                                             # Removes everything except alphabets
    text = " ".join([word for word in text.split() if word not in stop_words])  # Removes stopwords + spaces

    return text.lower()

def clean_word(word: str) -> str:
    delimeters = [char for char in " ,.-_—:;!?\""]

    for delimeter in delimeters:
        word = word.replace(delimeter, "")

    return word