# Imports
import Loader

# Main
if __name__ == "__main__":
    # # Loads, pre-processes and classifies the screenplays
    # classifications = Classifier.classify(sys.argv[1:])
    #
    # # Prints classifications to process
    # print(classifications.to_json(orient="records", indent=4))

    Loader.load_train_screenplays()

    """
    OUTPUT EXAMPLE:
    Title               |   GenrePercentages        
    "American Psycho"       {"Action": 22.43, "Adventure": 14.88 ... }
    """