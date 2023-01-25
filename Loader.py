# Imports
import pandas, json, pathlib, sys

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ScreenplayProcessor import extract_features
from Classifier import *

# Globals
genre_labels = json.load(open("Jsons/Genres.json"))


# Methods
def load_screenplay(file_path):
    screenplay_title = pathlib.Path(file_path).stem
    screenplay_text = open(file_path, "r", encoding="utf8").read()

    time.sleep(0.1)

    return extract_features(screenplay_title, screenplay_text)

def load_screenplays(file_paths):
    # TODO: FIX (1142 records, processes only 828)
    # Loads and processes each screenplay
    with ThreadPoolExecutor() as executor:
        screenplay_records = executor.map(load_screenplay, file_paths)

    return pandas.DataFrame(screenplay_records)


def load_genres():
    screenplays_info = pandas.read_json("Movie Script Info.json")
    genres_dict = {}

    # Builds a dictionary of screenplay genres by its title
    for offset, screenplay_info in screenplays_info.iterrows():
        genres_dict[screenplay_info["title"]] = screenplay_info["genres"]

    return pandas.DataFrame({"Title": genres_dict.keys(), "Actual Genres": genres_dict.values()})


# Main
if __name__ == "__main__":
    train_directory, train_pickle_file = f"./TrainScreenplays/", f"./Classifier/Screenplays.csv"
    pickle_path = pathlib.Path.cwd() / train_pickle_file
    train_file_names = os.listdir(train_directory)
    train_file_paths = [train_directory + file_name for file_name in train_file_names]
    genres = load_genres()

    if pathlib.Path.exists(pickle_path):
        train_screenplays_1 = pandas.read_pickle(train_pickle_file)
    else:
        train_screenplays_1 = pandas.merge(load_screenplays(train_file_paths), genres, on="Title")
        train_screenplays_1.to_pickle(train_pickle_file)

    portion_size = len(train_screenplays_1)
    print(f"Loaded {portion_size} train screenplays")

    remaining_file_names = train_file_names - list(train_screenplays_1["Title"])
    remaining_offsets = [train_file_names.index(name) for name in remaining_file_names]
    remaining_file_paths = [train_file_paths[offset] for offset in remaining_offsets]
    remaining_genres = [genres[offset] for offset in remaining_offsets]

    train_screenplays_2 = pandas.merge(load_screenplays(remaining_file_paths), remaining_genres, on="Title")
    train_screenplays_1.append(train_screenplays_2)
    train_screenplays_1.to_pickle(train_pickle_file)

    # # Loads and pre-processes screenplays to classify
    # screenplays = load_screenplays(sys.argv[1:])
    #
    # # Classifies the screenplays
    # classifications = classify(screenplays)
    #
    # # Prints classifications to process
    # print(classifications.to_json(orient="records", indent=4))

    """
    OUTPUT EXAMPLE:
    Title               |   GenrePercentages        
    "American Psycho"       {"Action": 22.43, "Adventure": 14.88 ... }
    """
