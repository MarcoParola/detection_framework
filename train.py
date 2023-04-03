import hydra
import os
from scripts.py.prepare_config import prepare_config


@hydra.main(config_path="./config/", config_name="config")
def train(cfg):

    config = prepare_config(cfg)

    if cfg.model == 'yolo':
        from ultralytics import YOLO
        model_path = os.path.join(cfg.project_path, cfg.config.actual_config_path, cfg.yolo.yolo_config.model_config)
        yolo_model_path = os.path.join(cfg.project_path, cfg.models.path, 'yolo', cfg.yolo.yolo_model)

        model = YOLO(model_path).load(yolo_model_path)  # build from YAML and transfer weights
        model.train(**config)   # Train the model

    if cfg.model == 'fasterRCNN':
        from detectron2.engine import DefaultTrainer
        trainer = DefaultTrainer(config)
        trainer.resume_or_load(resume=False)
        trainer.train()

    if cfg.model == "detr":
        '''
        main_path = os.path.join(cfg.project_path, "models", "detr", "main.py")
        batch_size = 2
        epochs = 10
        num_classes = 3
        dataset_file = "'coco'"
        coco_path = os.path.join(cfg.project_path, "data", "coco")
        output_dir = os.path.join(cfg.project_path, "outputs", "detr")
        device = "'cpu'"
        resume = "'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'" #detr-resnet50

        exec_command = f"{main_path}  \
                       --batch_size={batch_size} \
                       --epochs={epochs} \
                       --num_classes={num_classes} \
                       --dataset_file={dataset_file} \
                       --coco_path = {coco_path} \
                       --output_dir = {output_dir} \
                       --device = {device} \
                       --resume = {resume}"

        os.system("python " + exec_command)
        '''

if __name__ == '__main__':
    train()