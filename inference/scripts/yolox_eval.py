import re
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(2000)

model_path = '../tflite_models/float32/yolox.tflite'
img_path = './images/street.jpg'