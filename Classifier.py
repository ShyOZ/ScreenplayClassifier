# Imports
import pandas, pickle, os, time, pathlib

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from Setup import *

# Methods
def load_model():
    # Validates existence of pickle file
    pickle_path = pathlib.Path.cwd() / "Classifier/Pickle"
    if not pathlib.Path.exists(pickle_path):
        return train()

    # Reads model variables from pickle file
    pickle_file = open(pickle_path, "rb")
    model = pickle.load(pickle_file)
    pickle_file.close()

    return model

def save_model(model):
    # Writes model variables to pickle file
    pickle_file = open(pathlib.Path.cwd() / "Classifier/Pickle", "wb")
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

def train():
    # Loads train screenplays
    train_directory = f"./TrainScreenplays/"
    train_file_paths = [train_directory + file_name for file_name in os.listdir(train_directory)]
    train_screenplays = pandas.merge(load_screenplays(train_file_paths), load_genres(), on="Title")

    # Creates multi-label binary representation to the screenplays' genres
    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(train_screenplays["Actual Genres"])

    # Splits screenplays collection to train and validation
    x_train, x_validation, y_train, y_validation = train_test_split(train_screenplays["Text"].values, y,
                                                                    test_size=0.2, random_state=1000)

    # Extracts features from train screenplays
    vectorizer = TfidfVectorizer(max_df=0.8, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(x_train)
    x_validation = vectorizer.transform(x_validation)

    '''
        CLASSIFIERS HISTORY:
        OneVsRestClassifier(LogisticRegression()) -> 0.1205
    '''

    # TODO: Use feature-selection and ensembles with SGD/KNN classifiers
    # Classifies the test screenplays
    classifier = OneVsRestClassifier(LogisticRegression())
    classifier.fit(x_train, y_train)
    score = classifier.score(x_validation, y_validation)
    print("Accuracy: {:.4f}".format(score))

    # Saves model variables to file
    # save_model([vectorizer, classifier])

    return [vectorizer, classifier]

def classify(screenplays):
    # Loads classification model
    vectorizer, classifier = load_model()
    classifications_dict = {}
    classifications_complete = 0

    # Classifies each screenplay and organizes in dictionary
    for offset, screenplay in screenplays.iterrows():
        test_vector = vectorizer.transform([screenplay["Text"]])
        test_probabilities = sum(classifier.predict_proba(test_vector).tolist(), []) # Flattens the list
        test_percentages = probabilities_to_percentages(test_probabilities)

        classifications_dict[screenplay["Title"]] = test_percentages

        # Prints progress (for GUI to update progress)
        classifications_complete += 1
        print(classifications_complete)

        time.sleep(0.5) # seconds

    return pandas.DataFrame({"Title": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})