import cv2
import tensorflow as tf
import numpy as np
from common import preprocess_image


def ssdlite_mobiledet_inference(img_path):
    MODEL_PATH = '/home/pwl/Projects/tiny-tracking/tflite_models/quantized/ssdlite_mobiledet_dynamic_quant_.tflite'

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_img_shape = tuple(input_shape[1:3])
    input_type = input_details[0]['dtype']

    original_img, scaled_img = preprocess_image(
        img_path, input_img_shape, input_type)

    if input_type == np.float32:
        scaled_img /= 255.0

    interpreter.set_tensor(input_details[0]['index'], scaled_img)
    interpreter.invoke()
    bboxes = interpreter.get_tensor(output_details[0]['index']).squeeze()
    class_ids = interpreter.get_tensor(output_details[1]['index']).squeeze()
    scores = interpreter.get_tensor(output_details[2]['index']).squeeze()
    num_detections = interpreter.get_tensor(
        output_details[3]['index']).squeeze()

    # Perform non-max-suppresion and overwrite original outputs
    nms_idxs = tf.image.non_max_suppression(
        bboxes,
        scores,
        max_output_size=100,
        iou_threshold=0.4,
        score_threshold=0.25
    )
    bboxes = bboxes[nms_idxs]
    class_ids = class_ids[nms_idxs]
    scores = scores[nms_idxs]

    class_ids = class_ids + 1

    detections = list()

    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        y_min_ratio, x_min_ratio, y_max_ratio, x_max_ratio = bbox

        x_min = int(x_min_ratio * original_img.shape[1])
        y_min = int(y_min_ratio * original_img.shape[0])
        x_max = int(x_max_ratio * original_img.shape[1])
        y_max = int(y_max_ratio * original_img.shape[0])

        detections.append([x_min, y_min, x_max, y_max, score, class_id])

    return detections
