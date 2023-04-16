# Imports
import math
import time
import pandas
import Constants
import multiprocessing

from pathlib import Path
from datetime import datetime
from ScriptInfo import ScriptInfo
#from ScreenplayProcessor import extract_features
from concurrent.futures import ThreadPoolExecutor

# Methods
def load_screenplay(file_path):
    # Loads and processes a screenplay by its file path
    screenplay_title = Path(file_path).stem
    screenplay_text = open(file_path, "r", encoding="utf8").read()
    # screenplay_features = extract_features(screenplay_title, screenplay_text)

    time.sleep(0.01)

    return {"Title": screenplay_title, "Text": screenplay_text}

def load_train_screenplays():
    # Validates existence of required directories
    Constants.classifier_path.mkdir(parents=True, exist_ok=True)
    if not Path.exists(Constants.train_screenplays_directory):
        raise FileNotFoundError("TrainScreenplays directory not found.")

    # Retrieves the paths of the train screenplays left to load
    train_screenplays_paths = Constants.train_screenplays_paths
    if Path.exists(Constants.train_csv_path):
        trained_screenplays_titles = pandas.read_csv(Constants.train_csv_path, usecols=["Title"]).Title
        trained_screenplays_paths = map(lambda title: Constants.train_screenplays_directory / f"{title}.txt",
                                        trained_screenplays_titles)
        train_screenplays_paths = list(filter(lambda path: path not in trained_screenplays_paths,
                                              train_screenplays_paths))

    # Calculates amount of batches required
    batch_size = multiprocessing.cpu_count()
    batch_count = math.ceil(len(train_screenplays_paths) / batch_size)
    print(f"{datetime.now()}: Processing begun.")

    # Loads the train screenplays
    with ThreadPoolExecutor(batch_size) as executor:
        for i in range(batch_count):
            # Takes a batch of train screenplays file paths
            limit = min((i + 1) * batch_size, len(train_screenplays_paths))
            file_paths_batch = train_screenplays_paths[i * batch_size:limit]

            # Loads the screenplays batch using thread for each file path
            screenplay_threads = [executor.submit(load_screenplay, file_path) for file_path in file_paths_batch]
            screenplays_batch = [thread.result() for thread in screenplay_threads]

            # Appends the loaded batch to csv file
            screenplays_batch = pandas.DataFrame(screenplays_batch)
            screenplays_batch.to_csv(Constants.train_csv_path,
                                     mode="a",
                                     index=False,
                                     header=not Constants.train_csv_path.exists())

            print(f"{datetime.now()}: screenplays records were written to csv file.")

    # Merges the loaded train screenplays with their respective labels
    pandas.read_csv(Constants.train_csv_path).merge(load_genres()).to_csv(Constants.train_csv_path, index=False)

    print(f"{datetime.now()}: Processing ended.")

def load_test_screenplays(file_paths):
    # Loads the test screenplays using thread for each screenplay
    batch_size = len(file_paths)

    with ThreadPoolExecutor(batch_size) as executor:
        screenplay_threads = [executor.submit(load_screenplay, file_path) for file_path in file_paths]
        screenplay_records = [thread.result() for thread in screenplay_threads]

    return pandas.DataFrame(screenplay_records)

def load_genres():
    # Loads the genres labels for the train screenplays
    movie_info = ScriptInfo.schema().loads(Constants.movie_info_path.read_text(), many=True)

    genres_dict = {screenplay_info.title: list(screenplay_info.genres) for screenplay_info in movie_info}

    return pandas.DataFrame({"Title": genres_dict.keys(), "Genres": genres_dict.values()})