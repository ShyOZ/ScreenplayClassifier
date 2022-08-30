# Imports
import pandas as pd
import os, pathlib, sys

from Classifier import classify
from TextProcessor import *
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
        screenplays_dict[screenplay_title] = screenplay_text #process_text(screenplay_text)

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
    train_screenplays_df = pd.merge(read_train_screenplays(), read_genres(), on="Title")
    test_screenplays_df = read_test_screenplays(sys.argv[1:])
    concordances_dict = {}
    word_appearances_dict = {}

    classifications_df = classify(train_screenplays_df, test_screenplays_df)
    for _, classfication in classifications_df.iterrows():
        concordance, word_appearances = build_concordance_and_word_appearances(classfication["Text"])
        concordances_dict[classfication["Title"]] = [concordance]
        word_appearances_dict[classfication["Title"]] = [word_appearances]

    concordances_df = pd.DataFrame({"Title": concordances_dict.keys(), "Concordace": concordances_dict.values()})
    word_appearances_df = pd.DataFrame({"Title": word_appearances_dict.keys(),
                                        "Word Appearances": word_appearances_dict.values()})

    classifications_df = pd.merge(classifications_df, concordances_df, on="Title")
    classifications_df = pd.merge(classifications_df, word_appearances_df, on="Title")
    classifications_df = classifications_df.drop("Text", axis=1)

    """
    title               |   predicted genres    |   concordance                     |   appearances
    "american psycho"       [action, ...]           {'american': {0, 2018,...},...}      {'american': 8,...}
    """

    print(classifications_df.to_json(orient="records", indent=4))