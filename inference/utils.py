import cv2
import numpy as np

def preprocess_image(img_path, img_shape, input_type=np.float32):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img
    scaled_img = cv2.resize(img, img_shape)
    scaled_img = np.expand_dims(scaled_img, axis=0)
    if input_type == np.float32:
        scaled_img = scaled_img.astype(np.float32)
    elif input_type == np.uint8:
        scaled_img = scaled_img.astype(np.uint8)
    return scaled_img, original_img
