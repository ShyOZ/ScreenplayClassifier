# Imports
from nltk.corpus import wordnet

import Setup
from NLPUtilities import *

# Methods
def build_semantic_field(word):
    definitions = set(clean_text(synset.definition()) for synset in wordnet.synsets(word))
    semantic_field = []

    # Builds the word's semantic field by its definitions
    for definition in definitions:
        definition_pos_dict = get_parts_of_speech(definition)
        for words_collection in definition_pos_dict.values():
            semantic_field.extend(lemmatize_words(words_collection))

    return semantic_field

def build_genres_semantic_fields(genre_labels):
    semantic_fields_dict = {}

    # Builds a semantic field to each genre and organizes in dictionary
    for genre_label in genre_labels:
        genre_name = "Science Fiction" if genre_label == "SciFi" else genre_label
        genre_semantic_fields = []

        # Combines semantic fields from each word
        for word in get_words(genre_name):
            genre_semantic_fields.extend(build_semantic_field(word))

        semantic_fields_dict[genre_label] = set(genre_semantic_fields)

    return semantic_fields_dict