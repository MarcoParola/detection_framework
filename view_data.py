import hydra
import os
import fiftyone as fo
import cv2
import numpy as np

def convert_bbox_format(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = int((x - w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    x2 = int((x + w / 2) * img_width)
    y2 = int((y + h / 2) * img_height)
    return x1, y1, x2, y2


@hydra.main(config_path="./config/", config_name="config")
def view(cfg):
    if cfg.dataset == 'yolo':
        # Load the YOLO labels and images
        label_folder = os.path.join(cfg.datasets.path, 'yolo', cfg.datasets.dataset_type, 'labels')
        image_folder = os.path.join(cfg.datasets.path, 'yolo', cfg.datasets.dataset_type, 'images')
        output_folder = os.path.join(cfg.project_path, 'outputs', 'yolo', 'annotated_images_visualization')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in os.listdir(label_folder):
            # Load the label file
            label_file = os.path.join(label_folder, file_name)
            with open(label_file, 'r') as f:
                label_str = f.read()
            label_list = label_str.strip().split('\n')
            labels = []
            for label in label_list:
                label_parts = label.strip().split(' ')
                label_class = int(label_parts[0])
                label_bbox = list(map(float, label_parts[1:]))
                labels.append([label_class] + label_bbox)

            # Load the corresponding image
            img_file = os.path.join(image_folder, file_name.replace('.txt', '.jpg'))
            img = cv2.imread(img_file)

            # Draw the bounding boxes on the image
            for label in labels:
                label_class = label[0]
                bbox = label[1:]
                x1, y1, x2, y2 = convert_bbox_format(bbox, img.shape[1], img.shape[0])
                color = tuple(map(int, np.random.randint(0, 256, 3)))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(label_class), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Save the annotated image
            output_file = os.path.join(output_folder, file_name.replace('.txt', '.jpg'))
            cv2.imwrite(output_file, img)


    elif cfg.dataset == 'coco':
        img_path = os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path)
        coco_file = os.path.join(cfg.datasets.path, 'coco', cfg.datasets.dataset_type + '.json')

        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=img_path,
            labels_path=coco_file,
        )

        session = fo.launch_app(dataset)
        session.wait()

if __name__ == '__main__':
    view()