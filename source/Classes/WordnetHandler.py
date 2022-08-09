# Imports
from nltk.corpus import wordnet
#from typing import List

# Methods
def get_word_synsets(word: str) -> set:
    return set(wordnet.synsets(word))

def get_sentence_synsets(sentence: str) -> set:
    sentence_synsets = set()

    for word in sentence:
        sentence_synsets.add(tuple(get_word_synsets(word)))

    return sentence_synsets

def get_word_lemmas(word: str) -> set:
    word_synsets = get_word_synsets(word)
    word_lemmas = set()

    for synset in word_synsets:
        for lemma in synset.lemmas():
            word_lemmas.add(lemma.name())

    return word_lemmas

def get_sentence_lemmas(sentence: str) -> set:
    sentence_lemmas = set()

    for word in sentence:
        sentence_lemmas.add(tuple(get_word_lemmas(word)))

    return sentence_lemmas

# Main
if __name__ == "__main__":
    pass