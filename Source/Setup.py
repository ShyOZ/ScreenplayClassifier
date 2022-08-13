# Imports
import pandas as pd
import os, pathlib, re, sys

from Classifier import classify

# Methods
def clean_text(text: str) -> str:
    text = re.sub("\'", "", text)           # Removes backslash-apostrophe
    text = re.sub("[^a-zA-Z]", " ", text)   # Removes everything except alphabets
    text = ' '.join(text.split())           # Removes whitespaces

    return text.lower()

def read_screenplays() -> pd.DataFrame:
    screenplays_directory = "../Resources/Screenplays/"
    file_names = os.listdir(screenplays_directory)
    screenplays_dict = {}

    # Builds a dictionary of screenplay text by its title
    for i in range(len(file_names)):
        screenplay_title = pathlib.Path(file_names[i]).stem
        screenplay_text = open(f"{screenplays_directory}{screenplay_title}.txt", encoding="utf8").read()
        screenplays_dict[screenplay_title] = clean_text(screenplay_text)

    return pd.DataFrame({"Title": screenplays_dict.keys(), "Screenplay": screenplays_dict.values()})

def build_dataframe() -> pd.DataFrame:
    screenplays_df, genres_df = read_screenplays(), pd.read_csv("../Resources/Title_To_Genre.csv")
    records_count = len(screenplays_df)
    dataframe = pd.DataFrame({"ID": [i for i in range(records_count)]})
    dataframe = dataframe.join(pd.merge(screenplays_df, genres_df, on="Title"))

    # Removes un-classified screenplays


    return dataframe

# Main
if __name__ == "__main__":
    screenplay_args = sys.argv[1:]
    screenplays = build_dataframe()
    classifications = classify(screenplays)

    print(classifications)