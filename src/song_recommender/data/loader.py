import numpy as np
from PIL import Image
import cv2

def load_png_resized(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert('L')
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def load_raw_resized(path: str, image_size : int):
    return cv2.resize(np.load(path).astype(np.float32), 
                      (image_size, image_size), 
                      interpolation=cv2.INTER_AREA
                      )