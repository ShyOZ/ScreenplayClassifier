# Imports
#import nltk
#import sklearn

from TextAnalyzer import *

from sklearn.feature_extraction.text import TfidfVectorizer

from typing import List

# Methods
def classify_screenplay(screenplay_tokens: List[str], screenplay_corpus: dict) -> None:
    genre_list = ["Action", "Adventure", "Comedy", "Crime",
                  "Drama", "Family", "Fantasy", "Horror",
                  "Romance", "SciFi", "Thriller", "War"]

    print("WORDS:")
    for word in screenplay_corpus["Words"]:
        print(word)

    print("SENTENCES:")
    for sentence in screenplay_corpus["Sentences"]:
        print(sentence)

    # Gets the frequency of each word in the corpus
    '''tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(screenplay_corpus)
    print(tfidf_vectorizer.get_feature_names_out())
    print(X.toarray())'''

def process_screenplay(screenplay_path: str) -> None:
    with open(screenplay_path, "r") as screenplayFile:
        screenplay_text = screenplayFile.read()
        screenplay_tokens = get_tokens(screenplay_text)
        screenplay_corpus = get_corpus(screenplay_text)

        classify_screenplay(screenplay_tokens, screenplay_corpus)

# Main
if __name__ == "__main__":
    pass