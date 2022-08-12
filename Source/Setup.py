# Imports
import pandas as pd
import sys

from Classifier import classify
from typing import List

# Methods
def build_dataframe(file_paths: List[str]) -> pd.DataFrame:
    # TODO: Read all screenplays' text and metadata and arrange everything in dataframe
    # (columns: [UID, Name, Text, Predicted Genres, Actual Genres])

    return pd.DataFrame()

# Main
if __name__ == "__main__":
    screenplays_dataframe = build_dataframe(sys.argv[1:])
    classifications_dict = classify(screenplays_dataframe)

    print(classifications_dict)