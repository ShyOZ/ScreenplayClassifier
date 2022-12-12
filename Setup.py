# Imports
import pandas

from json import load
from pathlib import Path
from sys import argv

from Classifier import *
from TextProcessor import *

# Globals
genre_labels = load(open("Jsons/Genres.json"))

# Methods
def load_screenplays(file_paths):
    screenplays_dict = {}

    # Builds a dictionary of screenplay text by its title
    for file_path in file_paths:
        screenplay_title = Path(file_path).stem
        screenplay_text = open(file_path, "r", encoding="utf8").read()
        screenplays_dict[screenplay_title] = process_text(screenplay_text)

    return pandas.DataFrame({"Title": screenplays_dict.keys(), "Text": screenplays_dict.values()})

def load_genres():
    info_ds = pandas.read_json("Movie Script Info.json")
    genres_dict = {}

    # Builds a dictionary of screenplay genres by its title
    for offset, info in info_ds.iterrows():
        genres_dict[info["title"]] = info["genres"]

    return pandas.DataFrame({"Title": genres_dict.keys(), "Actual Genres": genres_dict.values()})

# Main
if __name__ == "__main__":
    test_screenplays = load_screenplays(argv[1:])
    model = load_model()
    classifications = classify(model, test_screenplays)

    """
    OUTPUT EXAMPLE:
    Title               |   GenrePercentages        
    "American Psycho"       {"Action": 22.43, "Adventure": 14.88 ... }
    """

    # Prints classifications to process
    print(classifications.to_json(orient="records", indent=4))