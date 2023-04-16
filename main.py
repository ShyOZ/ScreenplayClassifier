# Imports
import sys

import pandas

import classifier
import constants
import loader

TRAIN_MODE = True

# Main
if __name__ == "__main__":
    if TRAIN_MODE:
        # Loads, pre-processes and classifies the train screenplays
        loader.load_train_screenplays()
        classifier.create_nn_model()
        # train_screenplays = pandas.read_csv(Constants.train_csv_path)
        # # Tokenizes the screenplays' texts
        # screenplays_texts = train_screenplays["Text"]
        # print(screenplays_texts.info())
    else:
        # Loads, pre-processes and classifies the test screenplays
        classifications = classifier.classify(sys.argv[1:])

        # Prints classifications to process
        print(classifications.to_json(orient="records", indent=4))

    """
    OUTPUT EXAMPLE:
    Title               |   GenrePercentages        
    "American Psycho"       {"Action": 22.43, "Adventure": 14.88 ... }
    """