# Imports
from subprocess import run
from pathlib import Path

# Main
if __name__ == "__main__":
    run(["python", str(Path.cwd() / "main.py"), "./TrainScreenplays/12.txt"])
