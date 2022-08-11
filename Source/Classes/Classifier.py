# Imports
#import nltk
#import sklearn

from itertools import islice

from typing import List, Dict

from Source.Classes.TextAnalyzer import *
from Source.Classes import TokensClassifier
from Source.Classes import CorpusClassifier

# Methods
def process_screenplay(screenplay_path: str) -> List[str]:
    screenplay_text = open(screenplay_path, "r").read()
    screenplay_tokens = get_tokens(screenplay_text)
    screenplay_corpus = get_corpus(screenplay_text)

    return classify_screenplay(screenplay_tokens, screenplay_corpus)

def get_average_genres(dict1: Dict[str, List[float]], dict2: Dict[str, List[float]]) -> Dict[str, List[float]]:
    average_dict = {key: (dict1[key] + dict2[key]) / 2 for key in dict1.keys()}

    return {key: value for key, value in sorted(average_dict.items(), key=lambda item: item[1], reverse=True)}

def classify_screenplay(screenplay_tokens: List[str], screenplay_corpus: Dict[str, List[str]]) -> List[str]:
    genre_names = open("../Resources/Genres.txt", "r").read().splitlines()
    genres_by_tokens = TokensClassifier.classify(genre_names, screenplay_tokens)
    genres_by_corpus = CorpusClassifier.classify(genre_names, screenplay_corpus)
    screenplay_genres = get_average_genres(genres_by_tokens, genres_by_corpus)

    return dict(islice(screenplay_genres.items(), 3))