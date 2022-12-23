# Imports
import pandas, json, pathlib, sys

from concurrent.futures import ThreadPoolExecutor
from ScreenplayProcessor import process_screenplays
from Classifier import *

# Globals
genre_labels = json.load(open("Jsons/Genres.json"))

# Methods
def load_screenplays(file_paths):
    screenplays_dict = {}

    # Builds a dictionary of screenplay text by its title
    with ThreadPoolExecutor() as executor:
        for file_path in file_paths:
            screenplay_record = executor.submit(load_screenplay, file_path).result()
            screenplays_dict[screenplay_record[0]] = screenplay_record[1]

    # Builds a dataframe from the columns dictionary
    screenplays = pandas.DataFrame({"Title": screenplays_dict.keys(), "Text": screenplays_dict.values()})

    return process_screenplays(screenplays)

def load_screenplay(file_path):
    screenplay_title = pathlib.Path(file_path).stem
    screenplay_text = open(file_path, "r", encoding="utf8").read()

    return screenplay_title, screenplay_text

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
    print("Loading test screenplays...")
    start_time = time.time()

    screenplays = load_screenplays(sys.argv[1:]) # 50.778 seconds

    end_time = time.time()
    print(f"Loading complete [Total: {end_time - start_time} seconds]")

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