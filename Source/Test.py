import subprocess
from pathlib import Path

if __name__ == "__main__":
    subprocess.run(["python", str(Path.cwd() / "Setup.py"), "12", "Zootopia", "8MM"])