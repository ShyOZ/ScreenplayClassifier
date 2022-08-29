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

    text = re.sub("\W+", " ", text)                                                # Removes everything except alphabets
    text = " ".join([word for word in text.split() if word not in stop_words])  # Removes stopwords + spaces

    return text.lower()

def build_concordance_and_word_appearances(text: str) -> Tuple[Dict[str, Set[int]], Dict[str, int]]:
    concordance, word_appearances = defaultdict(set), defaultdict(lambda: 0)
    sentences = sent_tokenize(text)

    for i, sentence in enumerate(sentences):
        sentence = sentence.lower()
        words = re.split(" ,-_!.—:;?", sentence) # NEEDS FIXES
        for word in words:
            concordance[word].add(i)
            word_appearances[word] += words.count(word)

    return dict(concordance), dict(word_appearances)