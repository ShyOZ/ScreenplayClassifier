# Imports
import pandas as pd
import numpy as np
import nltk

from sklearn.datasets import make_multilabel_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from typing import List, Dict

# Methods
def classify(screenplay_dataframe: pd.DataFrame, screenplay_names : List[str]) -> List[str]:
    screenplays_to_classify = screenplay_dataframe.loc[screenplay_dataframe["Title"].isin(screenplay_names)]
    x, y = make_multilabel_classification(n_classes=12, random_state=0)
    multioutput_classifier = MultiOutputClassifier(LogisticRegression()).fit(x, y)

    print(multioutput_classifier.predict(x))

    return ["", "", ""]