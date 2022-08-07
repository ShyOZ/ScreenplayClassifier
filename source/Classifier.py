# Imports
from typing import List

import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize


def classify_screenplay(screenplay_tokens: List[str]) -> None:
    for token in screenplay_tokens:
        print(token)


# Methods
def process_screenplay(screenplay_path: str) -> None:
    with open(screenplay_path, "r") as screenplayFile:
        screenplay_text = screenplayFile.read()
        screenplay_tokens = word_tokenize(screenplay_text)

        classify_screenplay(screenplay_tokens)


# Main
if __name__ == '__main__':
    pass
