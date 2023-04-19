import hydra
import os
from ultralytics import YOLO
from scripts.py.prepare_config import prepare_config
from print_test import CustomTrainer
from models.detr.train import DetrTrainer


@hydra.main(config_path="./config/", config_name="config", version_base=None)
def train(cfg):

    config = prepare_config(cfg, "train")

    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, cfg.config.actual_config_path, cfg.yolo.yolo_config.model_config)
        yolo_model_path = os.path.join(cfg.project_path, cfg.models.path, 'yolo', cfg.yolo.yolo_model)

        model = YOLO(model_path).load(yolo_model_path)  # build from YAML and transfer weights
        model.train(**config)   # Train the model

    if cfg.model == 'fasterRCNN':
        trainer = CustomTrainer(config, cfg.training.early_stopping.patience)
        trainer.resume_or_load(resume=False)
        try:
            trainer.train()
        except Exception:
            print(f"\033[32mEarly stopping triggered \033[0m")

    if cfg.model == "detr":
        DetrTrainer(**config).main()



if __name__ == '__main__':
    train()