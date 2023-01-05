# Imports
import pandas, json, pathlib, sys

from concurrent.futures import ThreadPoolExecutor
from ScreenplayProcessor import extract_features
from Classifier import *

# Globals
genre_labels = json.load(open("Jsons/Genres.json"))

# Methods
def load_screenplays(file_paths):
    # Builds a screenplay record for each file path
    # TODO: FIX (1142 records, processes only 828)
    with ThreadPoolExecutor() as executor:
        screenplay_records = executor.map(load_screenplay, file_paths)

    return pandas.DataFrame(screenplay_records)

def load_screenplay(file_path):
    screenplay_title = pathlib.Path(file_path).stem
    screenplay_text = open(file_path, "r", encoding="utf8").read()

    time.sleep(0.1)

    return extract_features(screenplay_title, screenplay_text)

def load_genres():
    screenplays_info = pandas.read_json("Movie Script Info.json")
    genres_dict = {}

    # Builds a dictionary of screenplay genres by its title
    for offset, screenplay_info in screenplays_info.iterrows():
        genres_dict[screenplay_info["title"]] = screenplay_info["genres"]

    return pandas.DataFrame({"Title": genres_dict.keys(), "Actual Genres": genres_dict.values()})

# Main
if __name__ == "__main__":
    # Loads and pre-processes screenplays to classify
    screenplays = load_screenplays(sys.argv[1:])

    # Classifies the screenplays
    classifications = classify(screenplays)

    # Prints classifications to process
    print(classifications.to_json(orient="records", indent=4))

    """
    OUTPUT EXAMPLE:
    Title               |   GenrePercentages        
    "American Psycho"       {"Action": 22.43, "Adventure": 14.88 ... }
    """