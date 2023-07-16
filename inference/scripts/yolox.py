import cv2
import tensorflow as tf
import numpy as np
from common import get_labels, account_for_all_classes


def yolox_inference(img_path):
    LABEL_PATH = '../coco_labels_actual.txt'
    MODEL_PATH = '../../tflite_models/float32/yolox-tiny.tflite'
    labels = get_labels(LABEL_PATH)

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    img_shape = tuple(input_shape[1:3])
    input_type = input_details[0]['dtype']

    # Image preprocessing
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img
    scaled_img = cv2.resize(img, img_shape)
    scaled_img = np.expand_dims(scaled_img, axis=0)
    if input_type == np.float32:
        scaled_img = scaled_img.astype(np.float32)
    elif input_type == np.uint8:
        scaled_img = scaled_img.astype(np.uint8)
    else:
        raise ValueError(f'Unsupported input type: {input_type}')

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
        score_threshold=0.3
    )
    bboxes = bboxes[nms_idxs]
    class_ids = class_ids[nms_idxs]
    scores = scores[nms_idxs]

    class_ids = account_for_all_classes(class_ids)

    detections = list()

    # Convert coordinates to original image size and add to detections
    for bbox, score, class_id in zip(bboxes, scores, class_ids):

        x_min, y_min, x_max, y_max = bbox
        x_min_ratio = x_min / input_shape[1]
        y_min_ratio = y_min / input_shape[2]
        x_max_ratio = x_max / input_shape[1]
        y_max_ratio = y_max / input_shape[2]

        x_min = x_min_ratio * original_img.shape[1]
        y_min = y_min_ratio * original_img.shape[0]
        x_max = x_max_ratio * original_img.shape[1]
        y_max = y_max_ratio * original_img.shape[0]

        # print('[',x_min, y_min, x_max, y_max,']', f'{score:.2f}', label)
        detections.append([x_min, y_min, x_max, y_max, score, int(class_id)])

    return detections
