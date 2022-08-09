# Imports
from nltk.corpus import wordnet
from typing import List

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
def get_genre_synsets(genre: str) -> set:
    if genre == "SciFi":
        science, fiction = tuple(wordnet.synsets("Science")), tuple(wordnet.synsets("Fiction"))
        return set(science.__add__(fiction))

    return set(wordnet.synsets(genre))

def get_genres_semmantic_similarity(genre1: wordnet.synset, genre2: wordnet.synset) -> None:
    #TODO: COMPLETE (gets common hypernyms, hyponyms and such between 2 given genres)
    pass