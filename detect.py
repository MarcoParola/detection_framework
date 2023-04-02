import hydra
import os
import cv2

from ultralytics import YOLO

from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer

from scripts.py.prepare_config import prepare_config


@hydra.main(config_path="./config/", config_name="config")
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

        cfg.fastercnn.parameters.checkpoint_url = os.path.join(cfg.project_path, cfg.fastercnn.parameters.output_dir, cfg.fastercnn.fastercnn_model_path)
        config = prepare_config(cfg)

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
            v = Visualizer(img[:, :, ::-1])
            v = v.draw_instance_predictions(instances.to("cpu"))
            # Save the image with the bounding boxes
            output_path = os.path.join(output_folder, os.path.basename(d["file_name"]))
            cv2.imwrite(output_path, v.get_image()[:, :, ::-1])

if __name__ == '__main__':
    detect()