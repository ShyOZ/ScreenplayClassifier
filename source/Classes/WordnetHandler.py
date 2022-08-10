# Imports

from typing import Dict, List

from collections import defaultdict
from nltk.tag import *
from nltk.corpus import wordnet

from source.Classes.TextAnalyzer import get_words

# Important Definitions
'''
hypernyms: a word that names a broad category that includes other words.
            e.g.: "primate" is a hypernym for "chimpanzee" and "human".
hyponyms: a word whose meaning is included in that of another word.
            e.g.: "scarlet", "vermilion", and "crimson" are hyponyms of "red".           
meronyms: a part of something used to refer to the whole.
            e.g.: "faces" meaning "people", as in "they've seen a lot of faces come and go".
holonyms: a term that denotes a whole, a part of which is denoted by a second term.
            e.g.: "face" is a holonym of the word "eye".
antynyms: a word of opposite meaning.
            e.g.: "good" and "bad", "hot" and "cold", "up" and "down" etc.
'''

# Methods
def get_synsets_by_genre(genre : str) -> list:
    if genre == "Sci-Fi":
        return list(set(wordnet.synsets("Science")).union(set(wordnet.synsets("Fiction"))))

    return wordnet.synsets(genre)

def get_genre_synsets(genre_names: list) -> dict:
    return dict(zip(genre_names, [get_synsets_by_genre(genre) for genre in genre_names]))

def get_genres_definitions(genre_synsets: dict) -> dict:
    genre_definitions = {}

    for genre, synset_list in genre_synsets.items():
        genre_definitions[genre] = []
        for synset in synset_list:
            genre_definitions[genre].append(synset.definition())

    return genre_definitions

def get_parts_of_speech(text: str) -> Dict[str, List[str]]:
    temp_pos = defaultdict(lambda: list())

    for word, pos in pos_tag(get_words(text), tagset='universal'):
        temp_pos[pos].append(word)

    return {"Nouns": temp_pos["NOUN"], "Verbs": temp_pos["VERB"], "Adjectives": temp_pos["ADJ"], "Adverbs": temp_pos["ADV"]}

def get_genres_parts_of_speech(genre_definitions: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    genre_parts_of_speech = {}

    for genre, definitions in genre_definitions.items():
        genre_parts_of_speech[genre] = {"Nouns": [], "Verbs": [], "Adjectives": [], "Adverbs": []}
        for definition in definitions:
            definition_parts_of_speech = get_parts_of_speech(definition)
            for pos_name, pos_items in definition_parts_of_speech.items():
                genre_parts_of_speech[genre][pos_name].extend(pos_items)

        genre_parts_of_speech[genre]["Nouns"] = set(genre_parts_of_speech[genre]["Nouns"])
        genre_parts_of_speech[genre]["Verbs"] = set(genre_parts_of_speech[genre]["Verbs"])
        genre_parts_of_speech[genre]["Adjectives"] = set(genre_parts_of_speech[genre]["Adjectives"])
        genre_parts_of_speech[genre]["Adverbs"] = set(genre_parts_of_speech[genre]["Adverbs"])
        
    return genre_parts_of_speech