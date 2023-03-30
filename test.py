import hydra
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

@hydra.main(config_path="./config/", config_name="config")
def test(cfg):
    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, 'runs','detect', 'train8','weights','best.pt')
        model = YOLO(model_path)  # load a custom model
        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered

    if cfg.model == 'coco':
        config = get_cfg()
        config.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
        config.MODEL.WEIGHTS = os.path.join(cfg.project_path, 'outputs', 'coco_object_detection', 'model_final.pth')  # Set path model .pth
        config.MODEL.DEVICE = 'cpu'

        predictor = DefaultPredictor(config)

        images_path = os.path.join(cfg.datasets.path, "coco", "aug_images")
        output_dir = os.path.join(cfg.project_path, "outputs/coco_object_detection")

        test_dataset_name = "oralcancer_test"
        test_json_annot_path = os.path.join(cfg.datasets.path, "coco", "test.json")
        register_coco_instances(test_dataset_name, {}, test_json_annot_path, images_path)

        evaluator = COCOEvaluator(test_dataset_name, config, False, output_dir=output_dir)
        test_loader = build_detection_test_loader(config, test_dataset_name)
        inference_on_dataset(predictor.model, test_loader, evaluator)
    


if __name__ == '__main__':
    test()