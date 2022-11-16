# Imports
import pandas, os, pathlib, sys

from Classifier import *
from TextProcessor import *
from typing import List

# Methods
def read_train_screenplays() -> pandas.DataFrame:
    screenplays_directory = f"../Resources/Screenplays/"
    file_paths = os.listdir(screenplays_directory)
    screenplays_dict = {}

    # Builds a dictionary of screenplay text by its title
    for file_path in file_paths:
        screenplay_title = pathlib.Path(file_path).stem
        screenplay_text = open(f"{screenplays_directory}{file_path}", "r", encoding="utf8").read()
        screenplays_dict[screenplay_title] = process_text(screenplay_text)

    return pandas.DataFrame({"Title": screenplays_dict.keys(), "Text": screenplays_dict.values()})

def read_test_screenplays(file_paths: List[str]) -> pandas.DataFrame:
    screenplays_dict = {}

    # Builds a dictionary of screenplay text by its title
    for file_path in file_paths:
        screenplay_title = pathlib.Path(file_path).stem
        screenplay_text = open(file_path, "r", encoding="utf8").read()
        screenplays_dict[screenplay_title] = screenplay_text #process_text(screenplay_text)

    return pandas.DataFrame({"Title": screenplays_dict.keys(), "Text": screenplays_dict.values()})

def read_genres() -> pandas.DataFrame:
    genre_labels = open("../Resources/Genres.txt").read().splitlines()
    info_ds = pandas.read_json("../Resources/Movie Script Info.json")
    genres_dict = {}

    # Builds a dictionary of screenplay genres by its title
    for offset, info in info_ds.iterrows():
        genres_dict[info["title"]] = info["genres"]

    return pandas.DataFrame({"Title": genres_dict.keys(), "Actual Genres": genres_dict.values()})

# Main
if __name__ == "__main__":
    # Loads train and test screenplays
    train_screenplays = pandas.merge(read_train_screenplays(), read_genres(), on="Title")
    test_screenplays = read_test_screenplays(sys.argv[1:])

    # Trains the model and classifies the test screenplays
    classifier_variables = train(train_screenplays)
    classifications = classify(classifier_variables, test_screenplays)

    """
    OUTPUT EXAMPLE:
    Title               |   Genre Percentages        |  Concordance                     |   Word Appearances
    "american psycho"       {"Action"   : 23.67,
                             "Adventure": 12.92
                              ...}                      {"american": {0, 2018,...},...}     {"american": 8,...}
    """

    # Prints classifications to process
    print(classifications.to_json(orient="records", indent=4))