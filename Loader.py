# Imports
import time

import pandas, pathlib, json, sys, os

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ScreenplayProcessor import *
# from Classifier import *

# Globals
genre_labels = json.load(open("Jsons/Genres.json"))

# Methods
def load_screenplay(file_path):
    screenplay_title = pathlib.Path(file_path).stem
    screenplay_text = open(file_path, "r", encoding="utf8").read()
    screenplay_features = extract_features(screenplay_title, screenplay_text)

    time.sleep(0.01)

    print(f"{datetime.datetime.now()}: {screenplay_title} processed.")

    return screenplay_features

def load_screenplays():
    # Loads and processes each screenplay
    train_directory, train_csv_file = f"./TrainScreenplays/", f"./Classifier/TrainScreenplays.csv"
    csv_path = pathlib.Path.cwd() / train_csv_file
    train_file_names = os.listdir(train_directory)

    if pathlib.Path.exists(csv_path):
        loaded_screenplays = list(pandas.read_csv(csv_path)["Title"])
        train_file_names = [file_name for file_name in train_file_names
                            if pathlib.Path(file_name).stem not in loaded_screenplays]
        train_file_paths = [train_directory + file_name for file_name in train_file_names]

    batch_size = 10
    batch_count = len(train_file_paths) // batch_size
    print(f"{datetime.datetime.now()}: Processing begun.")

    # TODO: use futures threads (10 workers, batches of 10 screenplays)
    with ThreadPoolExecutor(batch_size) as executor:
        for i in range(batch_count):
            file_paths_batch = train_file_paths[:batch_size]

            screenplay_threads = [executor.submit(load_screenplay, file_path) for file_path in file_paths_batch]
            screenplays_batch = [thread.result() for thread in screenplay_threads]

            screenplays_batch = pandas.DataFrame(screenplays_batch)
            screenplays_batch.to_csv(csv_path, mode="a", index=False, header=False)
            print(f"{datetime.datetime.now()}: {batch_size} screenplay records were written to csv file.")

            train_file_paths = train_file_paths[batch_size:]

    print(f"{datetime.datetime.now()}: Processing ended.")

    # return pandas.DataFrame(screenplay_records)

def load_genres():
    screenplays_info = pandas.read_json("Movie Script Info.json")
    genres_dict = {}

    # Builds a dictionary of screenplay genres by its title
    for offset, screenplay_info in screenplays_info.iterrows():
        genres_dict[screenplay_info["title"]] = screenplay_info["genres"]

    return pandas.DataFrame({"Title": genres_dict.keys(), "Actual Genres": genres_dict.values()})

# Main
if __name__ == "__main__":
    load_screenplays()

    # file_path = sys.argv[1:][0]
    # screenplay_title = pathlib.Path(file_path).stem
    # screenplay_text = open(file_path, "r", encoding="utf8").read()
    #
    # print(f"{datetime.datetime.now()}: Processing begun.")
    # print(extract_features(screenplay_title, screenplay_text))
    # print(f"{datetime.datetime.now()}: Processing ended.")

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
