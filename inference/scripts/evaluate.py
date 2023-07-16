
import json
import os
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolox import yolox_inference
from matplotlib import pyplot as plt

ANNOTATION_PATH = '/home/pwl/Projects/tiny-tracking/COCO/annotations/instances_val2017.json'
IMG_PATH = '/home/pwl/Projects/tiny-tracking/COCO/val2017'
coco_ground_truth = COCO(ANNOTATION_PATH)

results = list()

for image_name in tqdm(os.listdir(IMG_PATH)):
    image_id = int(image_name.split('.')[0])
    dets = yolox_inference(os.path.join(IMG_PATH, image_name))
    for det in dets:
        bbox = det[0:4]
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        coco_bbox = [x_min, y_min, width, height]
        score = det[4]
        class_id = det[5]
        pred = {
            'image_id': image_id,
            'category_id': int(class_id),
            'bbox': [round(float(coord), 10) for coord in coco_bbox],
            'score': float(score)
        }
        results.append(pred)

with open('instances_yolox.json', 'w') as f:
    json.dump(results, f)

coco_predictions = coco_ground_truth.loadRes('instances_yolox.json')

coco_eval = COCOeval(coco_ground_truth, coco_predictions, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
