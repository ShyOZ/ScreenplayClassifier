import Classifier
import Constants

if __name__ == '__main__':
    # Loads, pre-processes and classifies the screenplays
    classifications = Classifier.classify([Constants.train_screenplays_paths[0]])

    # Prints classifications to process
    print(classifications.to_json(orient="records", indent=4))