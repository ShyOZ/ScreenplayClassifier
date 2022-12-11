# Imports
import os, time, pandas, pickle

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from Setup import genre_labels

# Methods
def load_pickle():
    pickle_path = Path.cwd() / "Classifier/Pickle"

    # Validation
    if not Path.exists(pickle_path):
        return [MultiLabelBinarizer(), TfidfVectorizer(min_df=0, lowercase=False), LogisticRegression()]

    # Reads model variables from pickle file
    pickle_file = open(pickle_path, "rb")
    binarizer, vectorizer, classifier = pickle.load(pickle_file)
    pickle_file.close()

    return [binarizer, vectorizer, classifier]

def save_pickle(binarizer, vectorizer, classifier):
    # Writes model variables to pickle file
    pickle_file = open(Path.cwd() / "Classifier/Pickle", "wb")
    pickle.dump([binarizer, vectorizer, classifier], pickle_file)
    pickle_file.close()

def probabilities_to_percentages(probabilities):
    probabilities_dict = dict(zip(genre_labels, probabilities))
    probabilities_dict = dict(sorted(probabilities_dict.items(), key=lambda item: item[1], reverse=True))
    sum_of_probabilities = sum(probabilities_dict.values())
    percentages_dict = {}

    # Converts each genre's probability to matching percentage
    for genre, probability in probabilities_dict.items():
        percentages_dict[genre] = (probability / sum_of_probabilities) * 100

    return percentages_dict

def train(train_screenplays):
    # Loads model variables from file
    binarizer, vectorizer, classifier = load_pickle()

    binarizer.fit(train_screenplays["Actual Genres"])
    y = binarizer.transform(train_screenplays["Actual Genres"])

    x_train, x_validation, y_train, y_validation = train_test_split(train_screenplays["Text"], y, test_size=0.2,
                                                                        random_state=42)

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_validation_tfidf = vectorizer.transform(x_validation)

    classifier.fit(x_train_tfidf, y_train)
    print("Accuracy:", classifier.score(x_train_tfidf, y_train)) # Accuracy: 0.14350453172205438 - MUST BE IMPROVED!

    return [binarizer, vectorizer, classifier]

def classify(model_variables, test_screenplays):
    binarizer, vectorizer, classifier = model_variables
    classifications_dict = {}
    classifications_complete = 0

    for offset, test_screenplay in test_screenplays.iterrows():
        test_vector = vectorizer.transform([test_screenplay["Text"]])
        test_probabilities = sum(classifier.predict_proba(test_vector).tolist(), []) # Flattens the list
        test_percentages = probabilities_to_percentages(test_probabilities)

        classifications_dict[test_screenplay["Title"]] = test_percentages

        # prints progress (for GUI to update progress)
        classifications_complete += 1
        print(classifications_complete)

        time.sleep(0.5) # Sleep for 0.5 seconds

    # Saves classifier's variables to file
    save_pickle(binarizer, vectorizer, classifier)

    return pandas.DataFrame({"Title": classifications_dict.keys(),
                             "GenrePercentages": classifications_dict.values()})