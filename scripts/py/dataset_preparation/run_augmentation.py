import json
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import cv2
import hydra


def augment_image_and_annotation(image, annotation):
    # Define the augmentation pipeline
    seq = iaa.Sequential([
        iaa.Multiply((0.95, 1.05)),  # Adjust brightness (95-105% of original)
        iaa.LinearContrast((0.95, 1.05)),  # Adjust contrast (95-105% of original)
        iaa.AddToHueAndSaturation((-10, 10)),  # Adjust hue and saturation (-10 to 10)
        iaa.Fliplr(0.5),  # Horizontally flip 50% of images
        iaa.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-10, 10),
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            mode="edge"
        )
    ])


    # Convert COCO format bounding boxes to imgaug format
    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=bb["bbox"][0], y1=bb["bbox"][1], x2=bb["bbox"][0] + bb["bbox"][2],
                    y2=bb["bbox"][1] + bb["bbox"][3])
        for bb in annotation["annotations"]
    ], shape=image.shape)

    # Apply augmentation
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    return image_aug, bbs_aug


def perform_augmentation(coco_data, images_input_path, images_output_path, initial_image_id, initial_annotation_id):
    new_images = []
    new_annotations = []
    initial_image_id = initial_image_id
    initial_annotation_id = initial_annotation_id

    for img_info in coco_data["images"]:
        img_path = os.path.join(images_input_path, img_info["file_name"])
        image = cv2.imread(img_path)

        img_annotations = {
            "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]],
            "image_id": img_info["id"]
        }

        for i in range(5):
            image_aug, bbs_aug = augment_image_and_annotation(image, img_annotations)

            new_images.append({
                "id": len(new_images) + initial_image_id + 1,
                "width": image_aug.shape[1],
                "height": image_aug.shape[0],
                "file_name": f"aug_{i}_{img_info['file_name']}"
            })

            # Convert imgaug bounding boxes back to COCO format
            annotations_aug = []
            for bb_idx, bb in enumerate(bbs_aug.bounding_boxes):
                x1_clipped = max(0, bb.x1)
                y1_clipped = max(0, bb.y1)
                x2_clipped = min(image_aug.shape[1], bb.x2)
                y2_clipped = min(image_aug.shape[0], bb.y2)
                width_clipped = x2_clipped - x1_clipped
                height_clipped = y2_clipped - y1_clipped

                if width_clipped > 0 and height_clipped > 0:
                  annotations_aug.append({
                      "id": len(new_annotations) + initial_annotation_id + 1 + bb_idx,
                      "image_id": len(new_images) + initial_image_id,
                      "category_id": img_annotations["annotations"][bb_idx]["category_id"],
                      "area": int(width_clipped * height_clipped),
                      "bbox": [round(float(x1_clipped),1), round(float(y1_clipped),1), round(float(width_clipped),1), round(float(height_clipped),1)],
                      "iscrowd": img_annotations["annotations"][bb_idx]["iscrowd"],
                      "isbbox": img_annotations["annotations"][bb_idx]["isbbox"],
                      "color": img_annotations["annotations"][bb_idx]["color"]
                  })

            new_annotations.extend(annotations_aug)

            # Save augmented image
            cv2.imwrite(os.path.join(images_output_path, f"aug_{i}_{img_info['file_name']}"), image_aug)

    return new_images,new_annotations

def save_augmented_annotations(coco_data, new_images, new_annotations, annotationsFileOutput):
    coco_data_augmented = coco_data.copy()
    coco_data_augmented["images"].extend(new_images)
    coco_data_augmented["annotations"].extend(new_annotations)

    with open(annotationsFileOutput, "w") as f:
        json.dump(coco_data_augmented, f)

def get_initial_id(coco_data, testAnnotationFile):
    with open(testAnnotationFile, "r") as f:
        test_coco_data = json.load(f)

    # Sort the 'images' field by their 'id'
    coco_data['images'] = sorted(coco_data['images'], key=lambda x: x['id'])
    test_coco_data['images'] = sorted(test_coco_data['images'], key=lambda x: x['id'])

    # Sort the 'annotations' field by their 'id'
    coco_data['annotations'] = sorted(coco_data['annotations'], key=lambda x: x['id'])
    test_coco_data['annotations'] = sorted(test_coco_data['annotations'], key=lambda x: x['id'])

    initial_image_id = max(coco_data["images"][-1]["id"], test_coco_data["images"][-1]["id"])
    initial_annotation_id = max(coco_data["annotations"][-1]["id"],test_coco_data["annotations"][-1]["id"])

    return initial_image_id, initial_annotation_id

@hydra.main(config_path="../../../config/", config_name="config")
def augmentation(cfg):
    annotationsFile = os.path.join(cfg.datasets.path, 'coco', 'train.json')
    testAnnotationFIle = os.path.join(cfg.datasets.path, 'coco', 'test.json')
    images_input_path = os.path.join(cfg.datasets.path, cfg.datasets.img_path)
    images_output_path =  os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path)

    with open(annotationsFile, "r") as f:
        coco_data = json.load(f)

    initial_image_id, initial_annotation_id = get_initial_id(coco_data, testAnnotationFIle)

    print(initial_image_id, initial_annotation_id)

    new_images, new_annotations = perform_augmentation(coco_data, images_input_path, images_output_path, initial_image_id, initial_annotation_id)

    save_augmented_annotations(coco_data, new_images, new_annotations, annotationsFile)


if __name__ == '__main__':
    augmentation()