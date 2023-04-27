import os
import hydra

from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from ultralytics import YOLO

from models.detr.train import DetrTrainer
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

        evaluator = COCOEvaluator(cfg.fastercnn.parameters.test_dataset_name, config, False,
                                  output_dir=cfg.fastercnn.parameters.output_dir)
        test_loader = build_detection_test_loader(config, cfg.fastercnn.parameters.test_dataset_name)
        inference_on_dataset(predictor.model, test_loader, evaluator)

    if cfg.model == 'detr':
        detr = DetrTrainer(**config)
        train_dataset, _, test_dataset = detr.create_dataset()
        train_dataloader = detr.data_loader(train_dataset, batch_size=config['train_batch_size'])
        test_dataloader = detr.data_loader(test_dataset, batch_size=config['test_batch_size'])
        model_path = os.path.join(config["output_path"], config["model_path"])
        model = detr.build_model(train_dataloader, test_dataloader)
        model = model.load_from_checkpoint(model_path, **config)
        detr.evaluation(test_dataset, test_dataloader, model)


if __name__ == '__main__':
    test()
