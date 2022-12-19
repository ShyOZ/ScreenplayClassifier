# Imports
import re, pandas, transformers

# Globals
emotion_pipeline = transformers.pipeline("sentiment-analysis", model="arpanghoshal/EmoRoBERTa")
emotion_labels = ["Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
                  "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment", "Excitement",
                  "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
                  "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise"]

# Methods
def get_emotions(text):
    emotions_dict = dict(zip(emotion_labels, [0 for label in emotion_labels]))
    emotions_scores = emotion_pipeline(text)

    # Organizes emotions and their scores in dictionary
    for emotion in emotions_scores:
        emotions_dict[emotion["label"].capitalize()] = emotion["score"]

    return emotions_dict

def process_screenplays(screenplays):
    # TODO: FIX (something when loading train screenplays fucks up)
    screenplays_emotions = [get_emotions(screenplay["Text"]) for offset, screenplay in screenplays.iterrows()]
    emotions_dict = dict(zip(screenplays_emotions[0].keys(), [[] for key in screenplays_emotions[0].keys()]))

    # Organizes emotions in lists
    for screenplay_emotions in screenplays_emotions:
        for emotion_label, emotion_score in screenplay_emotions.items():
            emotions_dict[emotion_label].append(emotion_score)

    # Removes the text column and adds a column for each emotion
    screenplays.drop("Text", axis=1)
    for emotion_label, emotion_scores in emotions_dict.items():
        screenplays[emotion_label] = emotion_scores

    return screenplays