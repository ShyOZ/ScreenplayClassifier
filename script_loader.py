from pathlib import Path
from typing import List, Collection, Tuple, Optional, Set

from script_info import ScriptInfo


class ScriptLoader:
    def __init__(self, folder_path: Path, script_iter: Collection[ScriptInfo] = None, is_train: bool = False):
        self.folderPath = folder_path
        self.files = folder_path.glob(f"*.txt")
        self.script_iter = iter(script_iter) if script_iter is not None else None
        self.is_train = is_train

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, str, Optional[str], Optional[Tuple[str]]]:
        try:
            if self.script_iter is not None:
                script = next(self.script_iter)
                text = (self.folderPath / f"{script.filename}.txt").read_text(encoding="utf8")
                genres = script.genres if self.is_train else None
                return text, script.filename, script.title, tuple(genres)
            else:
                file_path = next(self.files)
                filename = file_path.stem
                text = file_path.read_text(encoding="utf8")
                return text, filename, None, None

        except StopIteration:
            raise StopIteration
