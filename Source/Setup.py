# Imports
import pandas as pd
import os, pathlib, sys

from Classifier import classify
from TextProcessor import process_text
from typing import List

# Methods
def read_train_screenplays() -> pd.DataFrame:
    screenplays_directory = f"../Resources/Screenplays/"
    file_paths = os.listdir(screenplays_directory)
    screenplays_dict = {}

    # Builds a dictionary of screenplay text by its title
    for file_path in file_paths:
        screenplay_title = pathlib.Path(file_path).stem
        screenplay_text = open(f"{screenplays_directory}{file_path}", "r", encoding="utf8").read()
        screenplays_dict[screenplay_title] = process_text(screenplay_text)

    return pd.DataFrame({"Title": screenplays_dict.keys(), "Text": screenplays_dict.values()})

def read_test_screenplays(file_paths: List[str]) -> pd.DataFrame:
    screenplays_dict = {}

    for file_path in file_paths:
        screenplay_title = pathlib.Path(file_path).stem
        screenplay_text = open(file_path, "r", encoding="utf8").read()
        screenplays_dict[screenplay_title] = process_text(screenplay_text)

    return pd.DataFrame({"Title": screenplays_dict.keys(), "Text": screenplays_dict.values()})

def read_genres() -> pd.DataFrame:
    genre_labels = open("../Resources/Genres.txt").read().splitlines()
    info_ds = pd.read_json("../Resources/Movie Script Info.json")
    genres_dict = {}

    # Builds a dictionary of screenplay genres by its title
    for offset, info in info_ds.iterrows():
        genres_dict[info["title"]] = info["genres"]

    return pd.DataFrame({"Title": genres_dict.keys(), "Actual Genres": genres_dict.values()})

# Main
if __name__ == "__main__":
    train_screenplays = pd.merge(read_train_screenplays(), read_genres(), on="Title")
    test_screenplays = read_test_screenplays(sys.argv[1:])

    classifications = classify(train_screenplays, test_screenplays)
    print(classifications)