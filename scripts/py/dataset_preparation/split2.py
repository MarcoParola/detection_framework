import json
import hydra
import os
import random
import shutil


@hydra.main(config_path="../../../config/", config_name="config", version_base=None)
def split(cfg):
    train_annotation_file = os.path.join(cfg.datasets.path, 'coco', 'train.json')
    val_annotation_file = os.path.join(cfg.datasets.path, 'coco', 'val.json')
    aug_images_path = os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path)

    train_percentage = cfg.preproc.split_percentage

    with open(train_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Shuffle the list of images in the JSON file
    random.shuffle(coco_data['images'])

    num_train = int(train_percentage * len(coco_data['images']))

    train_images = coco_data['images'][:num_train]
    val_images = coco_data['images'][num_train:]

    train_annotations = []
    val_annotations = []

    # Copy the corresponding annotations to each set
    for ann in coco_data['annotations']:
        if ann['image_id'] in [x['id'] for x in train_images]:
            train_annotations.append(ann)
        elif ann['image_id'] in [x['id'] for x in val_images]:
            val_annotations.append(ann)

    # Create new COCO JSON files for each set
    train_coco_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': coco_data['categories']
    }

    val_coco_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': coco_data['categories']
    }

    # Write each set to its own COCO JSON file
    with open(train_annotation_file, 'w') as f:
        json.dump(train_coco_data, f)

    with open(val_annotation_file, 'w') as f:
        json.dump(val_coco_data, f)

    # Copy test images to coco/aug_images folder
    for image in train_images:
        image_path = os.path.join(cfg.datasets.path, cfg.datasets.img_path, image['file_name'])
        if os.path.exists(image_path):
            shutil.copy(image_path, aug_images_path)

    for image in val_images:
        image_path = os.path.join(cfg.datasets.path, cfg.datasets.img_path, image['file_name'])
        if os.path.exists(image_path):
            shutil.copy(image_path, aug_images_path)


if __name__ == '__main__':
    split()
