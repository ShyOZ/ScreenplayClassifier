# Imports
import nltk

from nltk.corpus import wordnet

from NLPUtilities import *

# Globals
genres_synonyms_dict = None

# Methods
def get_synonyms(word):
    synonyms = []

    for synset in wordnet.synsets(word):
        synonyms.extend([lemma.name() for lemma in synset.lemmas()])

    return set(synonyms)

def get_genres_synonyms(genre_labels):
    synonyms_dict = {}

    # Builds a semantic field to each genre and organizes in dictionary
    for genre_label in genre_labels:
        genre_name = "Science Fiction" if genre_label == "SciFi" else genre_label
        genre_semantic_fields = []

        # Combines semantic fields from each word
        for word in get_words(genre_name):
            genre_semantic_fields.extend(get_synonyms(word))

        synonyms_dict[genre_label] = set(genre_semantic_fields)

    return synonyms_dict