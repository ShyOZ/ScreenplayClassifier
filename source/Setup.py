# Imports
import sys
from Classes.Classifier import process_screenplay

# Main
if __name__ == '__main__':
    file_paths = sys.argv[1:]

    for file_path in file_paths:
        process_screenplay(file_path)
