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

def build_concordance_and_word_appearances(text: str) -> Tuple[Dict[str, Set[int]], Dict[str, int]]:
    concordance, word_appearances = defaultdict(set), defaultdict(lambda: 0)
    sentences = [sentence.lower() for sentence in sent_tokenize(text)]

    for i, sentence in enumerate(sentences):
        words = [clean_word(word) for word in sentence.split()]
        for word in words:
            concordance[word].add(i + 1) # lines in text are 1-based
            word_appearances[word] += 1

    return dict(concordance), dict(word_appearances)