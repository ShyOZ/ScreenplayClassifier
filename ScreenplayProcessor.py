# Imports
from NLPUtilities import *
from WordnetHandler import *
from Setup import *

# Methods
def process_screenplay(screenplay_text: str):
    screenplay_text = clean_text(screenplay_text)
    words = get_words(screenplay_text)

    # TODO: cross Semantic field of each genre label with each word and count occurrences per genre

    # Builds semantic field
    return screenplay_text
