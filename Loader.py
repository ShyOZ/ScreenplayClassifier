# Imports
import pandas, pathlib, json, sys, os

from datetime import datetime

from ScreenplayProcessor import *
# from Classifier import *

# Globals
genre_labels = json.load(open("Jsons/Genres.json"))

# Methods
def load_screenplays():
    # Loads and processes each screenplay
    train_directory, train_csv_file = f"./TrainScreenplays/", f"./Classifier/TrainScreenplays.csv"
    csv_path = pathlib.Path.cwd() / train_csv_file
    train_file_names = os.listdir(train_directory)
    train_file_paths = [train_directory + file_name for file_name in train_file_names]

    if pathlib.Path.exists(csv_path):
        loaded_screenplays = pandas.read_csv(csv_path)
        train_file_names = [file_name for file_name in train_file_names
                            if pathlib.Path(file_name).stem not in loaded_screenplays["Title"]]
        offsets_to_load = [train_file_names.index(file_name) for file_name in train_file_names]
        train_file_paths = [train_file_paths[offset] for offset in offsets_to_load]
    else:
        loaded_screenplays = pandas.DataFrame()
        screenplays_batch = []

    print(f"{datetime.datetime.now()}: Processing begun.")

    for file_path in train_file_paths[:3]:
        screenplay_title = pathlib.Path(file_path).stem
        screenplay_text = open(file_path, "r", encoding="utf8").read()
        screenplays_batch.append(extract_features(screenplay_title, screenplay_text))

        if len(screenplays_batch) == 10:
            loaded_screenplays.concat(pandas.DataFrame(screenplays_batch), ignore_index=True)
            loaded_screenplays.to_csv(csv_path)

            print(f"{datetime.datetime.now()}: 10 screenplay records were written to csv file.")
            screenplays_batch.clear()

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
    # load_screenplays()

    file_path = sys.argv[1:][0]
    screenplay_title = pathlib.Path(file_path).stem
    screenplay_text = open(file_path, "r", encoding="utf8").read()

    print(f"{datetime.datetime.now()}: Processing begun.")
    print(extract_features(screenplay_title, screenplay_text))
    print(f"{datetime.datetime.now()}: Processing ended.")

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
