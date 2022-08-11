# Imports
import sys

from os.path import basename
from Classes.Classifier import process_screenplay

# Main
if __name__ == "__main__":
    file_paths = sys.argv[1:]
    classifications_dict = {}

    for file_path in file_paths:
        classifications_dict[basename(file_path)] = process_screenplay(file_path)

    for screenplay, genres in classifications_dict.items():
        print(f"{screenplay}: {genres}")
