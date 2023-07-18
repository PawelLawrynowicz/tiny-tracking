import re
import numpy as np
import cv2
import tensorflow as tf


def get_coco_labels(labels_path):
    """
    Extract labels from COCO label file.
    """
    labels = dict()
    with open(labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number+1] = pair[0].strip()
    return labels


def get_colors_from_labels(labels):
    """
    Generate random colors for labels.
    """
    np.random.seed(2000)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="int32")
    return colors


def preprocess_image(source_path, target_shape, target_dtype):
    """
    Preprocess image for inference.
    """
    img = cv2.imread(source_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img
    scaled_img = cv2.resize(img, target_shape)
    scaled_img = np.expand_dims(scaled_img, axis=0)
    if target_dtype == np.float32:
        scaled_img = scaled_img.astype(np.float32)
    elif target_dtype == np.uint8:
        scaled_img = scaled_img.astype(np.uint8)
    return original_img, scaled_img


def fix_missing_labels(class_ids):
    """
    Fix missing labels in the COCO dataset. 
    Some detectors were trained on 80 classes when real amount of classes in COCO is 91.
    """
    missing_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
    for i in range(len(class_ids)):
        for label in missing_labels:
            if class_ids[i] > label:
                class_ids[i] = class_ids[i] + 1


def draw_bbox_with_label(original_img, x_min, y_min, x_max, y_max, score, label='none', color=(0, 255, 0)):
    cv2.rectangle(original_img, (x_min, y_min), (x_max, y_max), color, 6)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    detection_info = f'{label}: {score:.2f}'
    label_size = cv2.getTextSize(
        detection_info, font_face, font_scale, font_thickness)[0]
    label_x_min, label_y_min = x_min - 3, y_min - 3
    label_x_max, label_y_max = x_min + 3 + \
        label_size[0], y_min - 5 - label_size[1]
    cv2.rectangle(original_img, (label_x_min, label_y_min),
                  (label_x_max, label_y_max), color, -1)
    cv2.putText(original_img, detection_info, (x_min, y_min - 4),
                font_face, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
