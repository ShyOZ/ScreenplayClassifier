# Imports
import re, pandas

# Globals
from nltk import sent_tokenize
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline

emotion_pipeline = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
emotion_labels = ["Sadness", "Anger", "Fear", "Joy", "Surprise", "Love"]

# Methods
def get_screenplay_emotions(screenplay_text):
    screenplay_sentences = sent_tokenize(screenplay_text)
    sentences_count = len(screenplay_sentences)
    screenplay_emotions_dict = dict(zip(emotion_labels, [0 for emotion_label in emotion_labels]))
    sentences_emotions = []

    # Calculates average of each emotion and organizes in dictionary
    with ThreadPoolExecutor() as executor:
        for sentence in screenplay_sentences:
            sentences_emotions.append(executor.submit(get_emotions, sentence).result())

        for emotion_label in emotion_labels:
            screenplay_emotions_dict[emotion_label] = executor.submit(sum_emotion,
                                                                      sentences_emotions, emotion_label).result()
            screenplay_emotions_dict[emotion_label] /= sentences_count

    return screenplay_emotions_dict

def get_emotions(text):
    emotions_dict = dict(zip(emotion_labels, [0 for label in emotion_labels]))
    emotions_scores = sum(emotion_pipeline(text), []) # Flattens the list

    # Organizes emotions and their scores in dictionary
    for emotion in emotions_scores:
        emotions_dict[emotion["label"].capitalize()] = emotion["score"]

    return emotions_dict

def sum_emotion(emotions_records, emotion_label):
    return sum([emotions_record[emotion_label] for emotions_record in emotions_records])

def process_screenplays(screenplays):
    screenplays_texts = [screenplay["Text"] for offset, screenplay in screenplays.iterrows()]
    emotion_records = []

    # Adds column for each emotion to dataframe
    with ThreadPoolExecutor() as executor:
        for screenplays_text in screenplays_texts:
            emotion_records.append(executor.submit(get_screenplay_emotions, screenplays_text).result())

    for emotion_label in emotion_labels:
        screenplays[emotion_label] = [emotions_record[emotion_label] for emotions_record in emotion_records]

    # TODO: FURTHER FEATURE EXTRACTION (IF NECESSARY)

    # Drops the no longer required Text column
    screenplays.drop("Text", inplace=True, axis=1)

    return screenplays