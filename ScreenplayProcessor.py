# Imports
import re, pandas

# Globals
from transformers import pipeline

emotion_pipeline = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
emotion_labels = ["Sadness", "Anger", "Fear", "Joy", "Surprise", "Love"]

# Methods
def get_screenplay_emotions(screenplay_text):
    screenplay_sentences = screenplay_text.splitlines()
    sentences_count = len(screenplay_sentences)
    sentences_emotions = [get_emotions(sentence) for sentence in screenplay_sentences]
    screenplay_emotions_dict = dict(zip(emotion_labels, [0 for emotion_label in emotion_labels]))

    # Calculates average of each emotion and organizes in dictionary
    for sentence_emotions in sentences_emotions:
        for emotion_label in emotion_labels:
            screenplay_emotions_dict[emotion_label] += sentence_emotions[emotion_label]

    for emotion_label in emotion_labels:
        screenplay_emotions_dict[emotion_label] /= sentences_count

    return screenplay_emotions_dict

def get_emotions(text):
    emotions_dict = dict(zip(emotion_labels, [0 for label in emotion_labels]))
    emotions_scores = sum(emotion_pipeline(text), [])

    # Organizes emotions and their scores in dictionary
    for emotion in emotions_scores:
        emotions_dict[emotion["label"].capitalize()] = emotion["score"]

    return emotions_dict

def process_screenplays(screenplays):
    emotion_records = [get_screenplay_emotions(screenplay["Text"]) for _, screenplay in screenplays.iterrows()]

    # Adds column for each emotion to dataframe
    for emotion_label in emotion_labels:
        screenplays[emotion_label] = [emotions_record[emotion_label] for emotions_record in emotion_records]
    screenplays.drop("Text", inplace=True, axis=1)

    print(screenplays)

    return screenplays