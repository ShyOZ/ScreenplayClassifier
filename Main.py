# Imports
import sys
import Classifier
import Loader

# Main
if __name__ == "__main__":
    # Loads, pre-processes and classifies the train screenplays
    Loader.load_train_screenplays()

    # # Loads, pre-processes and classifies the test screenplays
    # classifications = Classifier.classify(sys.argv[1:])
    #
    # # Prints classifications to process
    # print(classifications.to_json(orient="records", indent=4))

    """
    OUTPUT EXAMPLE:
    Title               |   GenrePercentages        
    "American Psycho"       {"Action": 22.43, "Adventure": 14.88 ... }
    """