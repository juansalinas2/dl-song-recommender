from IPython.display import Audio, display
from PIL import Image

def play_audio(path):
    display(Audio(str(path)))

def show_image(path):
    display(Image.open(path))