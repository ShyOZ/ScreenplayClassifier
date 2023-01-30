# Imports
import time

import pandas, pathlib, json, sys, os

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ScreenplayProcessor import *
from Classifier import *

# Globals
genre_labels = json.load(open("Jsons/Genres.json"))

# Methods
def load_screenplay(file_path):
    # Loads and processes a screenplay by its file path
    screenplay_title = pathlib.Path(file_path).stem
    screenplay_text = open(file_path, "r", encoding="utf8").read()
    screenplay_features = extract_features(screenplay_title, screenplay_text)

    time.sleep(0.01)

    print(f"{datetime.datetime.now()}: {screenplay_title} processed.")

    return screenplay_features

def load_train_screenplays():
    # Loads and processes each screenplay
    train_directory, train_csv_file = f"./TrainScreenplays/", f"./Classifier/Train.csv"
    csv_path = pathlib.Path.cwd() / train_csv_file
    train_file_names = os.listdir(train_directory)
    train_file_paths = [train_directory + file_name for file_name in train_file_names]

    if pathlib.Path.exists(csv_path):
        loaded_screenplays = list(pandas.read_csv(csv_path)["Title"])
        train_file_names = [file_name for file_name in train_file_names
                            if pathlib.Path(file_name).stem not in loaded_screenplays]
        train_file_paths = [train_directory + file_name for file_name in train_file_names]

    batch_size = 50
    batch_count = len(train_file_paths) // batch_size
    print(f"{datetime.datetime.now()}: Processing begun.")

    with ThreadPoolExecutor(batch_size) as executor:
        for i in range(batch_count):
            file_paths_batch = train_file_paths[:batch_size]

            screenplay_threads = [executor.submit(load_screenplay, file_path) for file_path in file_paths_batch]
            screenplays_batch = [thread.result() for thread in screenplay_threads]

            screenplays_batch = pandas.DataFrame(screenplays_batch)
            screenplays_batch.to_csv(csv_path, mode="a", index=False, header=not pathlib.Path.exists(csv_path))
            print(f"{datetime.datetime.now()}: {batch_size} screenplay records were written to csv file.")

            train_file_paths = train_file_paths[batch_size:]

    print(f"{datetime.datetime.now()}: Processing ended.")

def load_test_screenplays(file_paths):
    # Loads and processes each screenplay
    batch_size = len(file_paths)

    with ThreadPoolExecutor(batch_size) as executor:
        screenplay_threads = [executor.submit(load_screenplay, file_path) for file_path in file_paths]
        screenplay_records = [thread.result() for thread in screenplay_threads]

    return pandas.DataFrame(screenplay_records)

def load_genres():
    # Builds a dictionary of screenplay genres by its title
    screenplays_info = pandas.read_json("Movie Script Info.json")
    genres_dict = {}

    for offset, screenplay_info in screenplays_info.iterrows():
        genres_dict[screenplay_info["title"]] = screenplay_info["genres"]

    return pandas.DataFrame({"Title": genres_dict.keys(), "Genres": genres_dict.values()})

# Main
if __name__ == "__main__":
    # Loads, pre-processes and classifies the screenplays
    classifications = classify(sys.argv[1:])

    # Prints classifications to process
    # print(classifications.to_json(orient="records", indent=4))

    """
    OUTPUT EXAMPLE:
    Title               |   GenrePercentages        
    "American Psycho"       {"Action": 22.43, "Adventure": 14.88 ... }
    """
