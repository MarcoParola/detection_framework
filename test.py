import hydra
import os

from ultralytics import YOLO
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from scripts.py.prepare_config import prepare_config

@hydra.main(config_path="./config/", config_name="config")
def test(cfg):
    config = prepare_config(cfg)
    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, cfg.yolo.parameters.output_dir, cfg.yolo.yolo_model_path)
        model = YOLO(model_path)  # load a custom model
        # Validate the model
        model.val(**config)  # no arguments needed, dataset and settings remembered

    if cfg.model == 'fasterRCNN':
        cfg.fastercnn.parameters.checkpoint_url = os.path.join(cfg.project_path, cfg.fastercnn.parameters.output_dir, cfg.fastercnn.fastercnn_model_path)
        config = prepare_config(cfg)

        predictor = DefaultPredictor(config)

        evaluator = COCOEvaluator(cfg.fastercnn.parameters.test_dataset_name, config, False, output_dir=cfg.fastercnn.parameters.output_dir)
        test_loader = build_detection_test_loader(config, cfg.fastercnn.parameters.test_dataset_name)
        inference_on_dataset(predictor.model, test_loader, evaluator)

if __name__ == '__main__':
    test()