# Imports
import random

from typing import List, Dict

#from sklearn.feature_extraction.text import TfidfVectorizer

# Methods
def classify(labels: List[str], tokens: List[str]) -> Dict[str, List[float]]:
    prob_dict = dict(zip(labels, [random.random() for i in range(len(labels))]))

    # TODO: COMPLETE
    # # Gets the frequency of each word in the corpus
    # tfidf_vectorizer = TfidfVectorizer()
    # X = tfidf_vectorizer.fit_transform(screenplay_corpus)
    # print(tfidf_vectorizer.get_feature_names_out())
    # print(X.toarray())

    return prob_dict