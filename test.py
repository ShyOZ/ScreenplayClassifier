# Imports
from subprocess import run
from pathlib import Path

import pandas

import classifier
import constants
import json

import loader
from imsdb_crawler import as_filename_compatible, standardize_title
from script_info import ScriptInfo

# Main
if __name__ == "__main__":
    run(["python", str(Path.cwd() / "main.py"), "./TrainScreenplays/12.txt"])
    # classifier.create_nn_model()
