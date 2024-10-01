import numpy as np
import cv2
import base64 

def decode_image(encoded_img) -> np.ndarray:
    np_img = np.fromstring(base64.b64decode(encoded_img), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    return img

def tif_to_ndarray(tif_path):
    img_array = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if img_array.dtype == np.uint16:
        img_array = (img_array / 256).astype(np.uint8)
    return img_array

def load_sample(enc_img: str):
    image = decode_image(enc_img)  # For decoding validation and evaluation files
    return image