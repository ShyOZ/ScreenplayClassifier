# Imports
import random
from typing import List, Dict

from Source.Classes.WordnetHandler import *

# Methods
def classify(labels: List[str], corpus: Dict[str, List[str]]) -> Dict[str, List[float]]:
    prob_dict = dict(zip(labels, [random.random() for i in range(len(labels))]))

    #genre_synsets_dict = dict(zip(labels, [get_genre_synsets(genre) for genre in labels]))
    #genre_definitions_dict = get_genres_definitions(genre_synsets_dict)
    #genre_pos_dict = get_genres_parts_of_speech(genre_definitions_dict)

    # TODO: COMPLETE

    return prob_dict