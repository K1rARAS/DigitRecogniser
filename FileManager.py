from pathlib import Path
from pickle import dump, load, HIGHEST_PROTOCOL
import sys

class FileM:
    def __init__(self):
        file_in = self.base_path_i() / "modelAIV1.pkl"
        dir_train = self.base_path_i() / "train"
        dir_out = self.base_path_o() / "ImagesForAI"
        self.paths = {"fi":file_in,"dt":dir_train,"do":dir_out}
        self.check_dir()

    def base_path_i(self):
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS).resolve()
        else:
            return Path(__file__).resolve().parent

    def check_dir(self):
        try:
            self.paths["dt"].mkdir(parents= True, exist_ok= True)
            self.paths["do"].mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print("Невозможно создать папки")
            return

    def base_path_o(self):
        if getattr(sys, "frozen", False):
            return Path(sys.executable).parent
        else:
            return Path(__file__).resolve().parent

    def get_path(self):
        return self.paths

    def scandir(self, path:Path = None):
        if path is None:
            path = self.paths["do"]
        result = []
        for p in path.glob("*"):
            result.append(str(p))
        return result

    def dump_AI(self, AI):
        path = self.paths["fi"]
        with path.open("wb") as f:
            dump(AI, f, protocol = HIGHEST_PROTOCOL)

    def load_AI(self):
        path = self.paths["fi"]
        with path.open("rb") as f:
            data = load(f)
            return data
FileM()