import cv2
import tensorflow as tf
import numpy as np
from common import preprocess_image, fix_missing_labels


def yolox_inference(img_path):
    MODEL_PATH = '/home/pwl/Projects/tiny-tracking/tflite_models/quantized/yolox_nano_dynamic_quant.tflite'

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_img_shape = tuple(input_shape[1:3])
    input_type = input_details[0]['dtype']

    original_img, scaled_img = preprocess_image(
        img_path, input_img_shape, input_type)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], scaled_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret outputs: [x_min, y_min, x_max, y_max, score, class_id]
    # coordinates are raw values of range [0, 415]
    bboxes = output_data[:, 0:4]
    scores = output_data[:, 4]
    class_ids = output_data[:, 5]

    # Non-maximum suppression
    nms_idxs = tf.image.non_max_suppression(
        bboxes,
        scores,
        max_output_size=100,
        iou_threshold=0.4,
        score_threshold=0.25,
    )
    bboxes = bboxes[nms_idxs]
    class_ids = class_ids[nms_idxs]
    scores = scores[nms_idxs]

    fix_missing_labels(class_ids)

    class_ids += 1

    detections = list()

    # Convert coordinates to original image size and add to detections
    for bbox, score, class_id in zip(bboxes, scores, class_ids):

        x_min, y_min, x_max, y_max = bbox
        x_min = int(x_min / scaled_img.shape[2] * original_img.shape[1])
        y_min = int(y_min / scaled_img.shape[1] * original_img.shape[0])
        x_max = int(x_max / scaled_img.shape[2] * original_img.shape[1])
        y_max = int(y_max / scaled_img.shape[1] * original_img.shape[0])

        # print('[',x_min, y_min, x_max, y_max,']', f'{score:.2f}', label)
        detections.append([x_min, y_min, x_max, y_max, score, int(class_id)])

    return detections
