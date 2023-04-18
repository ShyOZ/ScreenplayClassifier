# Imports
import json

from pathlib import Path

####################################################### Folders #######################################################

MOVIE_INFO_PATH = Path("Jsons/Movie Script Info.json")
GENRES_PATH = Path("Jsons/Genres.json")

TRAIN_SCREENPLAYS_PATH = Path("TrainScreenplays")
TEST_SCREENPLAYS_PATH = Path("TestScreenplays")

CLASSIFIER_PATH = Path("Classifier")
MODEL_PATH = CLASSIFIER_PATH / "Model"

######################################################## Files ########################################################

TRAIN_SCREENPLAYS_PATHS = list(TRAIN_SCREENPLAYS_PATH.glob("*.txt"))
TEST_SCREENPLAYS_PATHS = list(TEST_SCREENPLAYS_PATH.glob("*.txt"))

MODEL_PICKLE_PATH = CLASSIFIER_PATH / "Model.pkl"
TRAIN_CSV_PATH = CLASSIFIER_PATH / "Train.csv"
TEST_CSV_PATH = CLASSIFIER_PATH / "Test.csv"

###################################################### Variables ######################################################

TRAIN_MODE = False

GENRE_LABELS = json.loads(GENRES_PATH.read_text())
FEATURES_COUNT = 1000
DECISION_TREE_DEPTH = 7