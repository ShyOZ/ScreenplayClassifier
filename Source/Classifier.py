# Imports
import pandas as pd
import numpy as np
import nltk
import re
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import List, Dict

# Methods
def classify(screenplays: pd.DataFrame) -> Dict[int, List[str]]:
    genres = open("../Resources/Genres.txt").read().splitlines()
    # screenplay_text = open(screenplay_path, "r").read()

    # TODO: COMPLETE

    return ["", "", ""]