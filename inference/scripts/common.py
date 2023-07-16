import re


def get_labels(label_file_path):
    labels = dict()
    with open(label_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def account_for_all_classes(class_ids):
    """
    Account for all classes. COCO has 91 classes but the model only outputs 80. 11 of them are not used.
    Missing classes are: 12 (street sign), 26 (hat), 29 (shoe), 30 (eye glasses), 45 (plate), 66 (mirror), 68 (window), 69 (desk), 71 (door), 83 (blender), 91 (hair brush)
    """
    missing_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]

    for i in range(len(class_ids)):
        for label in missing_labels:
            if class_ids[i] > label:
                class_ids[i] = class_ids[i] + 1

    class_ids += 1
    return class_ids
    # ranges = [(1, 12), (12, 26), (26, 29), (29, 30), (30, 45),
    #           (45, 66), (66, 68), (68, 69), (69, 71), (71, 83), (83, 91)]
    # label_map = list()
    # for i, part in enumerate(ranges):
    #     for num in range(part[0], part[1]):
    #         label_map.append(i+1)

    # class_ids = [class_id + label_map[int(class_id)] for class_id in class_ids]
    # return class_ids
