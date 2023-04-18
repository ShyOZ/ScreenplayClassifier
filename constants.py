# Imports
import json

from pathlib import Path

####################################################### Folders #######################################################

movie_info_path = Path("Jsons/Movie Script Info.json")
genres_path = Path("Jsons/Genres.json")

train_screenplays_directory = Path("TrainScreenplays")
test_screenplays_directory = Path("TestScreenplays")

classifier_path = Path("Classifier")
model_path = classifier_path / "Model"

######################################################## Files ########################################################

train_screenplays_paths = list(train_screenplays_directory.glob("*.txt"))
test_screenplays_paths = list(test_screenplays_directory.glob("*.txt"))

model_pickle_path = classifier_path / "Model.pkl"
train_csv_path = classifier_path / "Train.csv"
test_csv_path = classifier_path / "Test.csv"

###################################################### Variables ######################################################

train_mode = False

genre_labels = json.loads(genres_path.read_text())
features_count = 1000
decision_tree_depth = 7