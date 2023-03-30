import hydra
import os
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import cv2



def get_train_cfg(config_file_path, model_final, num_classes, device, score_thresh_test):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_final  # Let training initialize from model zoo

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 4  # batch size

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Set number of classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test # Set threshold for this model
    cfg.MODEL.DEVICE = device  # CUDA

    return cfg

@hydra.main(config_path="./config/", config_name="config")
def detect(cfg):
    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, 'runs', 'detect', 'train', 'weights', 'best.pt')
        model = YOLO(model_path)  # load a custom model

        # define paths to input and output folders
        input_folder = os.path.join(cfg.datasets.path, 'yolo', 'test', 'images')
        output_folder = os.path.join(cfg.project_path, 'outputs', 'yolo', 'model_results_on_test')

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


    if cfg.model == 'coco':
        images_path = os.path.join(cfg.datasets.path, "coco", "aug_images")
        output_folder = os.path.join(cfg.project_path, 'outputs', 'coco', 'model_results_on_test')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        test_dataset_name = "oralcancer_test"
        test_json_annot_path = os.path.join(cfg.datasets.path, "coco", "test.json")

        register_coco_instances(test_dataset_name, {}, test_json_annot_path, images_path)

        config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        model_final = os.path.join(cfg.project_path, 'outputs', 'coco_object_detection', 'model_final.pth')  # Set path model .pth
        num_classes = 3
        device = "cuda"
        score_thresh_test = 0.5

        config = get_train_cfg(config_file_path, model_final, num_classes, device, score_thresh_test)

        predictor = DefaultPredictor(config)

        test_dataset_dicts = DatasetCatalog.get(test_dataset_name)

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