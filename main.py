# Imports
import sys
import classifier
import constants
import loader


# Main
if __name__ == "__main__":
    if constants.TRAIN_MODE:
        # Loads and pre-processes the train screenplays
        loader.load_train_screenplays()
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