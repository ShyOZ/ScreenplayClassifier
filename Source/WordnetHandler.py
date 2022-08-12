# Imports
from collections import defaultdict
from typing import Dict, List

from nltk.tag import *
from nltk.corpus import wordnet

from Source.TextProcessor import get_words

# Important Definitions
# synset: a group of synonymous words that express the same concept
# hypernyms: a word that names a broad category that includes other words.
#             e.g.: "primate" is a hypernym for "chimpanzee" and "human".
# hyponyms: a word whose meaning is included in that of another word.
#             e.g.: "scarlet", "vermilion", and "crimson" are hyponyms of "red".
# meronyms: a part of something used to refer to the whole.
#             e.g.: "faces" meaning "people", as in "they've seen a lot of faces come and go".
# holonyms: a term that denotes a whole, a part of which is denoted by a second term.
#             e.g.: "face" is a holonym of the word "eye".
# antynyms: a word of opposite meaning.
#             e.g.: "good" and "bad", "hot" and "cold", "up" and "down" etc.

# Methods
def get_genre_synsets(genre_name : str) -> list:
    genre_words = get_words(genre_name)
    genre_synsets = []

    for word in genre_words:
        genre_synsets.extend(wordnet.synsets(word))

    return genre_synsets

def get_genres_definitions(synsets_dict: dict) -> Dict[str, List[str]]:
    genre_definitions = defaultdict(lambda: list())

    for genre, genre_synsets in synsets_dict.items():
        for synset in genre_synsets:
            genre_definitions[genre].append(synset.definition())

    return genre_definitions

def get_parts_of_speech(text: str) -> Dict[str, List[str]]:
    temp_pos = defaultdict(lambda: list())

    for word, pos in pos_tag(get_words(text), tagset='universal'):
        temp_pos[pos].append(word)

    return {"Nouns": temp_pos["NOUN"], "Verbs": temp_pos["VERB"], "Adjectives": temp_pos["ADJ"], "Adverbs": temp_pos["ADV"]}