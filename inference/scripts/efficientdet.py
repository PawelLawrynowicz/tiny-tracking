import re
import cv2
import tensorflow as tf
import numpy as np


def efficientdet_inference(img_path):
    MODEL_PATH = '../../tflite_models/float32/efficientdet.tflite'

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    img_shape = tuple(input_shape[1:3])
    input_type = input_details[0]['dtype']

    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img, channels=3)
    original_img = img.numpy()
    if input_type == np.uint8:
        img = tf.image.convert_image_dtype(img, tf.uint8)
    elif input_type == np.float32:
        img = tf.image.convert_image_dtype(img, tf.float32)
    else:
        raise ValueError(f'Unsupported input type {input_type}')
    scaled_img = tf.image.resize(img, img_shape)
    scaled_img = scaled_img[tf.newaxis, :]

    interpreter.set_tensor(input_details[0]['index'], scaled_img)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).squeeze()
    bboxes = out[:, 1:5]
    scores = out[:, 5]
    class_ids = out[:, 6].astype(int)

    detections = list()

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        y_min_scaled, x_min_scaled, y_max_scaled, x_max_scaled = bbox

        x_min_ratio = x_min_scaled / img_shape[1]
        y_min_ratio = y_min_scaled / img_shape[0]
        x_max_ratio = x_max_scaled / img_shape[1]
        y_max_ratio = y_max_scaled / img_shape[0]

        x_min = int(x_min_ratio * original_img.shape[1])
        y_min = int(y_min_ratio * original_img.shape[0])
        x_max = int(x_max_ratio * original_img.shape[1])
        y_max = int(y_max_ratio * original_img.shape[0])

        detections.append([x_min, y_min, x_max, y_max, score, class_id])

    return detections
