# Imports
import sys
from Classifier import process_screenplay

# Main
if __name__ == '__main__':
    file_paths = sys.argv[1:]

    for filepath in file_paths:
        process_screenplay(filepath)
