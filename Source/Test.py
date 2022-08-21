import subprocess
from pathlib import Path

if __name__ == "__main__":
    subprocess.run(["python", str(Path.cwd() / "Setup.py"),
                    "../Resources/Screenplays/American Psycho.txt",
                    "../Resources/Screenplays/2012.txt",
                    "../Resources/Screenplays/Airplane.txt"])