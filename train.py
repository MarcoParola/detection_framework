import argparse
import os
import subprocess

import hydra
import os
from ultralytics import YOLO
from detectron2.engine import DefaultTrainer
from scripts.py.prepare_config import prepare_config
from print_test import CocoTrainer


@hydra.main(config_path="./config/", config_name="config")
def train(cfg):

    config = prepare_config(cfg)

    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, cfg.config.actual_config_path, cfg.yolo.yolo_config.model_config)
        yolo_model_path = os.path.join(cfg.project_path, cfg.models.path, 'yolo', cfg.yolo.yolo_model)

        model = YOLO(model_path).load(yolo_model_path)  # build from YAML and transfer weights
        model.train(**config)   # Train the model

    if cfg.model == 'fasterRCNN':
        trainer = CocoTrainer(config)
        trainer.resume_or_load(resume=False)
        trainer.train()

    if cfg.model == "detr":

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
    train()