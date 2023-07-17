import cv2
import tensorflow as tf
import numpy as np


def ssdlite_mobiledet_inference(img_path):
    MODEL_PATH = '../../tflite_models/float32/ssdlite_mobiledet.tflite'

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
        scaled_img = tf.image.resize(img, img_shape)
        scaled_img = scaled_img[tf.newaxis, :]
    else:
        raise ValueError(f'Input type {input_type} not supported')

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
        score_threshold=0.5
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
