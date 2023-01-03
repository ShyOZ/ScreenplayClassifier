# Imports
import pandas, pickle, os, time, pathlib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

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

def probabilities_to_percentages(probabilities):
    # Converts probabilities into sorted dictionary
    probabilities_dict = dict(zip(genre_labels, probabilities))
    probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    sum_of_probabilities = sum(probabilities_dict.values())
    percentages_dict = {}

    # Converts each genre's probability to matching percentage
    for genre, probability in probabilities_dict.items():
        percentages_dict[genre] = (probability / sum_of_probabilities) * 100

    return percentages_dict

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

    # Splits train screenplays into features (x) and targets (t), and splitting x into train and validation
    binarizer = MultiLabelBinarizer()
    t = binarizer.fit_transform(train_screenplays["Actual Genres"])
    x = train_screenplays.drop(["Title", "Actual Genres"], axis=1)
    # x_train, x_validation, y_train, y_validation = train_test_split(x, t, test_size=0.2, random_state=1000)

    # Builds classifier and predicts its score (best score: 0.9903)
    classifier = DecisionTreeClassifier().fit(x, t)
    score = classifier.score(x, t)
    print("Accuracy: {:.4f}".format(score))

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
        test_vector = "something" #vectorizer.transform([screenplay[1:]])
        # TODO: FIX (wrong argument passed to predict_proba)
        test_probabilities = sum(classifier.predict_proba(test_vector).tolist(), []) # Flattens the list
        test_percentages = probabilities_to_percentages(test_probabilities)

        classifications_dict[screenplay["Title"]] = test_percentages

        # Prints progress (for GUI to update progress)
        classifications_complete += 1
        print(classifications_complete)

        time.sleep(0.5) # seconds

    return pandas.DataFrame({"Title": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})