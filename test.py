import hydra
import os
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def get_train_cfg(config_file_path, model_final, num_classes, device, score_thresh_test, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_final  # Let training initialize from model zoo

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 4  # batch size

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Set number of classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test # Set threshold for this model
    cfg.MODEL.DEVICE = device  # CUDA

    cfg.OUTPUT_DIR = output_dir



    return cfg

@hydra.main(config_path="./config/", config_name="config")
def test(cfg):
    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, 'runs','detect', 'train','weights','best.pt')
        model = YOLO(model_path)  # load a custom model
        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered

    if cfg.model == 'coco':
        images_path = os.path.join(cfg.datasets.path, "coco", "aug_images")
        output_dir = os.path.join(cfg.project_path, "outputs/coco_object_detection")

        test_dataset_name = "oralcancer_test"
        test_json_annot_path = os.path.join(cfg.datasets.path, "coco", "test.json")
        register_coco_instances(test_dataset_name, {}, test_json_annot_path, images_path)

        val_dataset_name = "oralcancer_val"
        val_json_annot_path = os.path.join(cfg.datasets.path, "coco", "val.json")
        register_coco_instances(val_dataset_name, {}, val_json_annot_path, images_path)



        config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        model_final = os.path.join(cfg.project_path, 'outputs', 'coco_object_detection', 'model_final.pth')  # Set path model .pth
        num_classes = 3
        device = "cuda"
        score_thresh_test = 0.5

        config = get_train_cfg(config_file_path, model_final, num_classes, device, score_thresh_test, output_dir, )


        predictor = DefaultPredictor(config)

        evaluator = COCOEvaluator(test_dataset_name, config, False, output_dir=output_dir)
        test_loader = build_detection_test_loader(config, test_dataset_name)
        inference_on_dataset(predictor.model, test_loader, evaluator)
    


if __name__ == '__main__':
    test()