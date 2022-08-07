import subprocess
from pathlib import Path

if __name__ == "__main__":
    subprocess.run(["python", str(Path.cwd() / "Setup.py"), "../Resources/test.txt"])
    print(str(Path.cwd() / "Setup.py"))
