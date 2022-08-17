import subprocess
from pathlib import Path

if __name__ == "__main__":
    subprocess.run(["python", str(Path.cwd() / "Setup.py"),
                    "../Resources/Screenplays/Test/American Psycho.txt",
                    "../Resources/Screenplays/Test/2012.txt",
                    "../Resources/Screenplays/Test/Airplane.txt"])