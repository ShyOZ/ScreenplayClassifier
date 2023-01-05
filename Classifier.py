# Imports
import pandas, pickle, os, time, pathlib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import log_loss, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, LeavePOut, cross_validate
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from Setup import *

# Methods
def load_model():
    # Validates existence of pickle file
    pickle_path = pathlib.Path.cwd() / "Classifier/Model.pkl"
    if not pathlib.Path.exists(pickle_path):
        return create_model()

    # Reads model variables from pickle file
    pickle_file = open(pickle_path, "rb")
    model = pickle.load(pickle_file)
    pickle_file.close()

    return model

def save_model(model):
    # Writes model variables to pickle file
    pickle_file = open(pathlib.Path.cwd() / "Classifier/Model.pkl", "wb")
    pickle.dump(model, pickle_file)
    pickle_file.close()

def create_model():
    # Retrieves train screenplays
    train_directory, train_pickle_file = f"./TrainScreenplays/", f"./Classifier/Screenplays.pkl"
    pickle_path = pathlib.Path.cwd() / train_pickle_file
    if pathlib.Path.exists(pickle_path):
        train_screenplays = pandas.read_pickle(train_pickle_file)
    else:
        train_file_paths = [train_directory + file_name for file_name in os.listdir(train_directory)]
        train_screenplays = pandas.merge(load_screenplays(train_file_paths), load_genres(), on="Title")

        train_screenplays.to_pickle(train_pickle_file)

    print(f"Loaded {len(train_screenplays)} train screenplays")
    # Splits train screenplays into features (x) and targets (t), and splitting x into train and validation
    binarizer = MultiLabelBinarizer()
    t = binarizer.fit_transform(train_screenplays["Actual Genres"])
    x = train_screenplays.drop(["Title", "Actual Genres"], axis=1)
    x_train, x_validation, y_train, y_validation = train_test_split(x, t, test_size=0.2, random_state=1)

    # TODO: Improve the base model
    # Builds classifier and predicts its F1 score (best score: 0.3482)
    base_model = OneVsRestClassifier(DecisionTreeClassifier())
    classifier = base_model.fit(x_train, y_train)

    y_predictions = classifier.predict(x_validation)
    score = f1_score(y_validation, y_predictions, average="micro")
    print("F1-Score: {:.4f}".format(score))

    # Saves model variables to file
    save_model(classifier)

    return classifier

def classify(screenplays):
    # Loads classification model
    classifier = load_model()
    classifications_dict = {}
    classifications_complete = 0

    # Classifies each screenplay and organizes in dictionary
    for offset, screenplay in screenplays.iterrows():
        test_vector = [screenplay.values[1:]]
        test_probabilities = classifier.predict_proba(test_vector)[0]
        predicted_genres_count = sum(test_probabilities)
        test_percentages = [(prob / predicted_genres_count) * 100 for prob in test_probabilities]

        classifications_dict[screenplay["Title"]] = dict(zip(genre_labels, test_percentages))

        # Prints progress (for GUI to update progress)
        classifications_complete += 1
        print(classifications_complete)

        time.sleep(0.5) # seconds

    return pandas.DataFrame({"Title": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})