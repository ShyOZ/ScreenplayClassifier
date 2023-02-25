# Imports
import time
import numpy
import pandas
import pickle
import Constants
from Loader import load_test_screenplays
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Methods
def save_model(model):
    # Writes model variables to pickle file
    pickle_file = open(Constants.model_pickle_path, "wb")
    pickle.dump(model, pickle_file)
    pickle_file.close()

def create_model():
    # Creates a classification model
    train_screenplays = pandas.read_csv(Constants.train_csv_path)

    t = MultiLabelBinarizer().fit_transform(train_screenplays["Genres"])
    x = train_screenplays.drop(["Title", "Genres"], axis=1)

    x_train, x_validation, y_train, y_validation = train_test_split(x, t, test_size=0.2, random_state=1)

    # Builds classifier and predicts its accuracy score (best score: 0.0)
    # TOOD: IMPROVE THE base model with hyper_parameters
    base_model = OneVsRestClassifier(LogisticRegression())
    classifier = base_model.fit(x_train, y_train)

    y_predictions = classifier.predict(x_validation)
    score = accuracy_score(y_validation, y_predictions)
    print("Accuracy: {:.4f}".format(score))

    # Saves model variables to file
    save_model(classifier)

    return classifier

def load_model():
    # Validates existence of pickle file
    if not Constants.model_pickle_path.exists():
        return create_model()

    # Reads model variables from pickle file
    pickle_file = open(Constants.model_pickle_path, "rb")
    model = pickle.load(pickle_file)
    pickle_file.close()

    return model

def probabilities_to_percentages(probabilities):
    probabilities_dict = dict(zip(Constants.genre_labels, probabilities))
    probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    sum_of_probabilities = sum(probabilities)
    percentages_dict = {}

    # Converts each genre's probability to matching percentage
    for genre, probability in probabilities_dict.items():
        percentages_dict[genre] = (probability / sum_of_probabilities) * 100

    return percentages_dict

def classify(file_paths):
    # Loads classification model
    screenplays = load_test_screenplays(file_paths)
    classifier = load_model()
    classifications_dict = {}
    classifications_complete = 0

    # TODO: Figure out the test_probabilities
    for offset, screenplay in screenplays.iterrows():
        features_vector = [x[0] for x in numpy.array(screenplay[1:]).reshape(-1, 1)]
        test_probabilities = classifier.predict_proba([features_vector])
        print(test_probabilities)

        # test_percentages = probabilities_to_percentages(test_probabilities)
        # classifications_dict[file_paths[offset]] = test_percentages

        # Prints progress (for GUI to update progress)
        # classifications_complete += 1
        # print(classifications_complete)

        time.sleep(0.5)  # seconds

    return pandas.DataFrame({"FilePath": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})
