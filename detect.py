import json

import hydra
import os
import cv2

from ultralytics import YOLO

from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from transformers import DetrFeatureExtractor

from models.detr.detr import Detr
from scripts.py.prepare_config import prepare_config
from models.detr.prediction import visualize_predictions

from PIL import Image


@hydra.main(config_path="./config/", config_name="config", version_base=None)
def detect(cfg):
    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, cfg.yolo.parameters.output_dir, cfg.yolo.yolo_model_path)
        model = YOLO(model_path)  # load a custom model

        # define paths to input and output folders
        input_folder = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.test, cfg.datasets.img_path)
        output_folder = os.path.join(cfg.project_path, cfg.yolo.yolo_detect_output_path)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # loop over each image in the input folder
        for image_name in os.listdir(input_folder):
            # read image
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)

            # detect objects and get bounding boxes
            res = model(image)
            res_plotted = res[0].plot()

            # save image with bounding boxes
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, res_plotted)

    if cfg.model == 'fasterRCNN':
        output_folder = os.path.join(cfg.project_path, cfg.fastercnn.fastercnn_detect_output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cfg.fastercnn.parameters.checkpoint_url = os.path.join(cfg.project_path, cfg.fastercnn.parameters.output_dir,
                                                               cfg.fastercnn.fastercnn_model_path)
        config = prepare_config(cfg, 'test')

        predictor = DefaultPredictor(config)

        test_dataset_dicts = DatasetCatalog.get(cfg.fastercnn.parameters.test_dataset_name)
        # Loop over each image in the test dataset
        for d in test_dataset_dicts:
            # Load the image
            img = cv2.imread(d["file_name"])
            # Use the predictor to generate predictions for the image
            outputs = predictor(img)
            # Get the predicted instances with the highest confidence scores
            instances = outputs["instances"]
            scores = instances.scores.tolist()
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]
            instances = instances[indices]
            # Visualize the predictions on the image
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.fastercnn.parameters.test_dataset_name))
            v = v.draw_instance_predictions(instances.to("cpu"))
            # Save the image with the bounding boxes
            output_path = os.path.join(output_folder, os.path.basename(d["file_name"]))
            cv2.imwrite(output_path, v.get_image()[:, :, ::-1])


    if cfg.model == "detr":
        # define paths to input and output folders
        input_folder = os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path)
        output_folder = os.path.join(cfg.project_path, cfg.detr.detr_detect_output_path)

        test_annotation_file = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.test)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(test_annotation_file, 'r') as f:
            test_data = json.load(f)

        # Define the model and the feature extractor
        model_path = os.path.join(os.path.join(cfg.project_path, cfg.detr.parameters.output_dir),
                                  cfg.detr.detr_model_path)
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        model = Detr(num_labels=cfg.datasets.n_classes)
        model = model.load_from_checkpoint(model_path)
        model.eval()

        # Apply detection to each test image
        for image_info in test_data["images"]:
            image_name = image_info["file_name"]
            image_path = os.path.join(input_folder, image_name)

            img = Image.open(image_path)

            encoding = feature_extractor(img, return_tensors="pt")
            encoding.keys()

            outputs = model(**encoding)
            visualize_predictions(img, outputs, output_folder, image_name, cfg.datasets.class_name)


if __name__ == '__main__':
    detect()
