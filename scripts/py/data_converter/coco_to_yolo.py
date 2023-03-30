import json
import os
import hydra
import shutil


def parse_coco_json(coco_json_file):
    with open(coco_json_file, "r") as f:
        data = json.load(f)
    return data


def create_class_dict(data):
    class_dict = {}
    for category in data["categories"]:
        class_id = category["id"]
        class_name = category["name"]
        class_dict[class_id] = class_name
    return class_dict


def convert_bbox_format(bbox, width, height):
    x, y, w, h = bbox
    x_center = x + (w / 2)
    y_center = y + (h / 2)
    return [x_center / width, y_center / height, w / width, h / height]


def save_class_names(class_dict, class_file):
    with open(class_file, "w") as f:
        for class_id in sorted(class_dict):
            f.write(f"{class_dict[class_id]}\n")


@hydra.main(config_path="../../../config/", config_name="config")
def coco_to_yolo(cfg):
    coco_json_file = os.path.join(cfg.datasets.path, 'coco', cfg.datasets.dataset_type) + ".json"
    label_folder = os.path.join(cfg.datasets.path, 'yolo', cfg.datasets.dataset_type, 'labels')
    images_folder = os.path.join(cfg.datasets.path, 'yolo', cfg.datasets.dataset_type, 'images')
    class_file = os.path.join(cfg.datasets.path, 'yolo', 'classes.txt')

    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    data = parse_coco_json(coco_json_file)
    class_dict = create_class_dict(data)

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        class_id = annotation["category_id"]
        bbox = annotation["bbox"]

        image_info = [x for x in data["images"] if x["id"] == image_id][0]
        width, height = image_info["width"], image_info["height"]
        image_name = image_info["file_name"].rsplit(".", 1)[0]

        yolo_bbox = convert_bbox_format(bbox, width, height)

        label_file = os.path.join(label_folder, f"{image_name}.txt")

        with open(label_file, "a") as f:
            f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

    save_class_names(class_dict, class_file)

    # Copy test images to coco/aug_images folder
    for image in data['images']:
        image_path = os.path.join(cfg.datasets.path, 'coco', 'aug_images', image['file_name'])
        if os.path.exists(image_path):
            shutil.copy(image_path, images_folder)

    print(f"{cfg.datasets.dataset_type} -> Number of images moved: {len(data['images'])}")




if __name__ == '__main__':
    coco_to_yolo()
