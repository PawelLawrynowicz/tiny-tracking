
import json
import os
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolox import yolox_inference
from yolov8 import yolov8_inference
from ssdmobilenet import ssdlite_mobilenet_inference
from ssdmobiledet import ssdlite_mobiledet_inference
from efficientdet import efficientdet_inference

from matplotlib import pyplot as plt

ANNOTATION_PATH = '/home/pwl/Projects/tiny-tracking/COCO/instances_val2017.json'
IMG_PATH = '/home/pwl/Projects/tiny-tracking/COCO/val2017'
coco_ground_truth = COCO(ANNOTATION_PATH)
results = list()

models = {
    'yolox': yolox_inference,
    'yolov8': yolov8_inference,
    'ssdmobilenet': ssdlite_mobilenet_inference,
    'ssdmobiledet': ssdlite_mobiledet_inference,
    'efficientdet': efficientdet_inference
}
for model_name, model_function in models.items():

    chosen_model = model_name
    print(chosen_model)

    for image_name in tqdm(os.listdir(IMG_PATH)):
        image_id = int(image_name.split('.')[0])
        dets = models[chosen_model](os.path.join(IMG_PATH, image_name))
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

    OUT_FILE = f'dq_predictions_{chosen_model}.json'

    with open(OUT_FILE, 'w') as f:
        json.dump(results, f)

    coco_predictions = coco_ground_truth.loadRes(OUT_FILE)

    coco_eval = COCOeval(coco_ground_truth, coco_predictions, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    all_precision = coco_eval.eval['precision']
    pr_50 = all_precision[0, :, 0, 0, 2]  # IoU@0.5
    pr_75 = all_precision[5, :, 0, 0, 2]  # IoU@0.75
    pr_95 = all_precision[9, :, 0, 0, 2]  # IoU@0.95

    x = np.arange(0, 1.01, 0.01)
    ax = plt.subplot(111)

    ax.plot(x, pr_50, label='IoU@0.5')
    ax.plot(x, pr_75, label='IoU@0.75')
    ax.plot(x, pr_95, label='IoU@0.95')
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.grid()
    ax.spines[['right', 'top']].set_visible(False)

    # Area under curves
    area_5 = np.trapz(pr_50, x)
    area_7 = np.trapz(pr_75, x)
    area_9 = np.trapz(pr_95, x)
    print("AOC: ", area_5, area_7, area_9)

    plt.xlabel('Czułość')
    plt.ylabel('Precyzja')
    plt.legend()

    plt.savefig(f'dq_prc_{chosen_model}.png', dpi=300, bbox_inches='tight')
