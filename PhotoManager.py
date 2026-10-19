from FileManager import FileM
from PIL import Image

class PhotoM:
    def __init__(self):
        self.fm = FileM()

    def get_image(self, path):
        try:
            img = Image.open(path).convert("L").resize((28, 28), Image.Resampling.LANCZOS)
            pixels = img.getdata()
            massive = [p/255.0 for p in pixels]
            return massive
        except FileNotFoundError:
            print(f"Картинка не найдена — {path}")
            return None