# Imports
import re
import nltk
import spacy
import text2emotion

from datetime import datetime
from textblob import TextBlob
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor

# Globals
spacy = spacy.load("en_core_web_sm")
times_of_day = ["Daytime", "Nighttime"]
time_periods = ["Past", "Present", "Future"]
emotion_labels = ["Happy", "Angry", "Surprise", "Sad", "Fear"]

# Methods
def get_time_of_day(text):
    daytime_counter, nighttime_counter = 0, 0
    dusk_time, dawn_time = datetime.time(18), datetime.time(6)
    time_entities = [ent for ent in spacy(text).ents if ent.label_ == "TIME"]

    # Checks all time expressions in the text
    time_expressions = re.findall(r"^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$", " ".join(time_entities))
    time_expressions = [datetime.datetime.strptime(time_str, "%H:%M").time() for time_str in time_expressions]

    daytime_counter += len([time for time in time_expressions if dawn_time <= time < dusk_time])
    nighttime_counter += len(time_expressions) - daytime_counter

    # Checks all day/night related expressions in the text
    daytime_expressions = ["day", "dawn", "morning", "noon", "sun"]
    days_of_week = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
    nighttime_expressions = ["night", "dusk", "evening", "twilight"]
    words = clean_words(TextBlob(text).words)

    for expression in daytime_expressions:
        daytime_counter += len([w for w in words if (expression in w) and (w not in days_of_week)])
    for expression in nighttime_expressions:
        nighttime_counter += len([w for w in words if expression in w])

    return {"Time of Day": "Daytime" if daytime_counter > nighttime_counter else "Nighttime"}

def get_time_period(text):
    past_counter, present_counter, future_counter = 0, 0, 0
    date_entities = [ent for ent in spacy(text).ents if ent.label_ == "DATE"]

    # Counts years in 3 time periods: Past, Present, Future
    years = [int(year) for year in re.findall(r"[0-9]{4}", " ".join(date_entities))]
    past_counter += len([year for year in years if year < 1990])
    future_counter += len([year for year in years if year > datetime.date.today().year])
    past_counter += len(years) - (past_counter + future_counter)

    # Determines the time period
    time_period = "Present"
    if all(past_counter > counter for counter in [present_counter, future_counter]):
        time_period = "Past"
    elif all(present_counter > counter for counter in [past_counter, future_counter]):
        time_period = "Present"
    elif all(future_counter > counter for counter in [past_counter, present_counter]):
        time_period = "Future"

    return {"Time Period": time_period}

def get_emotions(text):
    # Extracts probabilities for 5 emotions: Happy, Angry, Surprise, Sad, Fear
    return text2emotion.get_emotion(text)

def clean_words(words):
    # Cleans the words from punctuation
    words = [re.sub(r"[^\w\s]", "", w) for w in words]

    # Cleans the words from stopwords (e.g.: an, of, is, at, by...)
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w.lower() not in stop_words]

    return words

def extract_features(screenplay_title, screenplay_text):
    features_dict = {"Title": screenplay_title}
    feature_methods = [get_emotions, get_time_period, get_time_of_day]

    # Extracts features by the screenplay text
    with ThreadPoolExecutor(len(feature_methods)) as executor:
        feature_threads = [executor.submit(method, screenplay_text) for method in feature_methods]
        features_list = [thread.result() for thread in feature_threads]

    # Joins all features into one dictionary
    for features_batch in features_list:
        features_dict.update(features_batch)

    return features_dict