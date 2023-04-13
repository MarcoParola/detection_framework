import subprocess

import hydra
import os

import torch
from ultralytics import YOLO
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from scripts.py.prepare_config import prepare_config

@hydra.main(config_path="./config/", config_name="config", version_base=None)
def test(cfg):
    config = prepare_config(cfg, "test")

    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, cfg.yolo.parameters.output_dir, cfg.yolo.yolo_model_path)
        model = YOLO(model_path)  # load a custom model
        # Validate the model
        model.val(**config)  # no arguments needed, dataset and settings remembered

    if cfg.model == 'fasterRCNN':
        predictor = DefaultPredictor(config)

        evaluator = COCOEvaluator(cfg.fastercnn.parameters.test_dataset_name, config, False, output_dir=cfg.fastercnn.parameters.output_dir)
        test_loader = build_detection_test_loader(config, cfg.fastercnn.parameters.test_dataset_name)
        inference_on_dataset(predictor.model, test_loader, evaluator)

    if cfg.model == 'detr':
        process = subprocess.Popen(config.split(), stdout=subprocess.PIPE)

        # Read the output of the subprocess while it is running
        while True:
            output = process.stdout.readline()
            if not output:
                break
            print(output.decode().strip())

        # Wait for the subprocess to finish
        process.wait()


if __name__ == '__main__':
    test()