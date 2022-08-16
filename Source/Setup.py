# Imports
import pandas as pd
import os, pathlib, sys

from Classifier import classify
from TextProcessor import process_text

# Methods
def read_screenplays() -> pd.DataFrame:
    screenplays_directory = "../Resources/Screenplays/"
    file_names = os.listdir(screenplays_directory)
    screenplays_dict = {}

    # Builds a dictionary of screenplay text by its title
    for i in range(len(file_names)):
        screenplay_title = pathlib.Path(file_names[i]).stem
        screenplay_text = open(f"{screenplays_directory}{screenplay_title}.txt", "r", encoding="utf8").read()
        screenplays_dict[screenplay_title] = process_text(screenplay_text)

    return pd.DataFrame({"Title": screenplays_dict.keys(), "Text": screenplays_dict.values()})

def read_genres() -> pd.DataFrame:
    genre_labels = open("../Resources/Genres.txt").read().splitlines()
    genres_df = pd.read_csv("../Resources/Title_To_Genre.csv")
    genres_dict = {}

    # Builds a dictionary of screenplay genres by its title
    for offset, classification in genres_df.iterrows():
        screenplay_title = classification["Title"]
        actual_genres = [label for label in genre_labels if int(classification[label]) == 1]

        if len(actual_genres) > 0:
            genres_dict[screenplay_title] = actual_genres

    return pd.DataFrame({"Title": genres_dict.keys(), "Actual Genres": genres_dict.values()})

# Main
if __name__ == "__main__":
    train_screenplays = pd.merge(read_screenplays(), read_genres(), on="Title")
    test_screenplays = train_screenplays.loc[train_screenplays["Title"].isin(sys.argv[1:])]

    classifications = classify(train_screenplays, test_screenplays)
    print(classifications)