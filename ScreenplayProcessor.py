# Imports
import re, pandas

# Globals
from transformers import pipeline

emotion_pipeline = pipeline(task="sentiment-analysis", model="arpanghoshal/EmoRoBERTa")
emotion_labels = ["Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
                  "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment", "Excitement",
                  "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
                  "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise"]

# Methods
def get_screenplay_emotions(screenplay_text):
    screenplay_sentences = screenplay_text.splitlines()
    sentences_count = len(screenplay_sentences)
    chunks_emotions = [get_emotions(sentence) for sentence in screenplay_sentences]
    screenplay_emotions_dict = dict(zip(emotion_labels, [0 for emotion_label in emotion_labels]))

    # Calculates average of each emotion and organizes in dictionary
    for chunk_emotion in chunks_emotions:
        for emotion_label in emotion_labels:
            screenplay_emotions_dict[emotion_label] += chunk_emotion[emotion_label]

    for emotion_label in emotion_labels:
        screenplay_emotions_dict[emotion_label] /= sentences_count

    print(screenplay_emotions_dict)

    return screenplay_emotions_dict

def get_emotions(text_chunk):
    emotions_dict = dict(zip(emotion_labels, [0 for label in emotion_labels]))
    emotions_scores = emotion_pipeline(text_chunk)

    # Organizes emotions and their scores in dictionary
    for emotion in emotions_scores:
        emotions_dict[emotion["label"].capitalize()] = emotion["score"]

    return emotions_dict

def process_screenplays(screenplays):
    emotion_records = [get_screenplay_emotions(screenplay["Text"]) for screenplay in screenplays]

    # print(emotion_records)

    # Adds column for each emotion to dataframe
    # for emotion_label in emotion_labels:
    #     screenplays[emotion_label] = [emotions_record[emotion_label] for emotions_record in emotion_records]
    # screenplays.drop("Text", axis=1)

    print(screenplays)

    return screenplays