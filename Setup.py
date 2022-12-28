# Imports
import pandas, json, pathlib, sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ScreenplayProcessor import extract_features
from Classifier import *

# Globals
genre_labels = json.load(open("Jsons/Genres.json"))

# Methods
def load_screenplays(file_paths):
    screenplay_records = []

    print("Loading screenplays...")
    start_time = time.time()

    # Builds a screenplay record for each file path
    with ThreadPoolExecutor(max_workers=100) as executor:
        screenplay_records = executor.map(load_screenplay, file_paths)

    end_time = time.time()
    print(f"Screenplays load complete [Total: {end_time - start_time} seconds].")

    return pandas.DataFrame(screenplay_records)

def load_screenplay(file_path):
    screenplay_title = pathlib.Path(file_path).stem
    screenplay_text = open(file_path, "r", encoding="utf8").read()

    return extract_features(screenplay_title, screenplay_text)

def load_genres():
    info_ds = pandas.read_json("Movie Script Info.json")
    genres_dict = {}

    # Builds a dictionary of screenplay genres by its title
    for offset, info in info_ds.iterrows():
        genres_dict[info["title"]] = info["genres"]

    return pandas.DataFrame({"Title": genres_dict.keys(), "Actual Genres": genres_dict.values()})

# Main
if __name__ == "__main__":
    # Loads and pre-processes screenplays to classify
    screenplays = load_screenplays(sys.argv[1:])

    print(screenplays)
    # Classifies the screenplays
    # create_model()
    # classifications = classify(screenplays)

    # Prints classifications to process
    # print(classifications.to_json(orient="records", indent=4))

    """
    OUTPUT EXAMPLE:
    Title               |   GenrePercentages        
    "American Psycho"       {"Action": 22.43, "Adventure": 14.88 ... }
    """