import subprocess
from pathlib import Path

if __name__ == "__main__":
    subprocess.run(["python", str(Path.cwd() / "Setup.py"), "../Resources/sample_text.txt"])
    print(str(Path.cwd() / "Setup.py"))
