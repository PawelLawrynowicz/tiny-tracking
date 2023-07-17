import cv2
import tensorflow as tf
import numpy as np
from common import get_labels, account_for_all_classes


def yolov8_inference(img_path):
    MODEL_PATH = '../../tflite_models/float32/yolov8n.tflite'

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
    else:
        img = tf.image.convert_image_dtype(img, tf.float32)
    scaled_img = tf.image.resize(img, img_shape)
    scaled_img = scaled_img[tf.newaxis, :]

    interpreter.set_tensor(input_details[0]['index'], scaled_img)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).squeeze()

    bboxes = out[:4, :].T

    for bbox in bboxes:
        bbox_ratios = bbox / scaled_img.shape[1]
        x_c, y_c, w, h = bbox_ratios
        bbox[0] = (x_c - w / 2) * original_img.shape[1]
        bbox[1] = (y_c - h / 2) * original_img.shape[0]
        bbox[2] = (x_c + w / 2) * original_img.shape[1]
        bbox[3] = (y_c + h / 2) * original_img.shape[0]

    bboxes = bboxes.astype(np.int32)
    all_scores = out[4:, :]
    class_ids = np.argmax(all_scores, axis=0)
    scores = all_scores[class_ids, np.arange(all_scores.shape[1])]

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

    class_ids = account_for_all_classes(class_ids)

    detections = list()

    for bbox, class_id, score in zip(bboxes, class_ids, scores):
        x_min, y_min, x_max, y_max = bbox
        detections.append([x_min, y_min, x_max, y_max, score, int(class_id)])

    return detections
