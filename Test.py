# Imports
from subprocess import run
from pathlib import Path

if __name__ == "__main__":
    run(["python", str(Path.cwd() / "Setup.py"), "./template.txt"])