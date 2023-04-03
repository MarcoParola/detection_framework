import json
import hydra
import os

from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances


def get_yolo_configuration(cfg):
    config = {
        "project": os.path.join(cfg.project_path, cfg.yolo.parameters.output_dir),
        "data": os.path.join(cfg.project_path, cfg.config.actual_config_path, cfg.yolo.yolo_config.data_config),
        "lr0": cfg.training.lr,
        "epochs": cfg.training.epochs,
        "batch": cfg.training.batch,
        "patience": cfg.training.early_stopping.patience,
        "optimizer": cfg.training.optimizer,
        "device":  cfg.yolo.parameters.device,
        "workers": cfg.training.workers,
        "imgsz": cfg.training.img_size
    }
    return config


def get_num_images(json_path):
    with open(json_path, "r") as f:
        dataset = json.load(f)
    image_ids = [image['id'] for image in dataset['images']]
    return len(image_ids)


def get_fastercnn_configuration(cfg):
    images_path = os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path)
    train_json_annot_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.train)
    val_json_annot_path = os.path.join(cfg.datasets.path,  cfg.datasets.datasets_path.coco.val)
    test_json_annot_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.test)

    # Register the dataset for the model usages
    try:
        register_coco_instances(cfg.fastercnn.parameters.train_dataset_name, {}, train_json_annot_path, images_path)
        register_coco_instances(cfg.fastercnn.parameters.val_dataset_name, {}, val_json_annot_path, images_path)
        register_coco_instances(cfg.fastercnn.parameters.test_dataset_name, {}, test_json_annot_path, images_path)
    except AssertionError:
        pass

    # Get number of training images
    num_train_images = get_num_images(train_json_annot_path)
    config = get_cfg()
    config.merge_from_file(model_zoo.get_config_file(cfg.fastercnn.parameters.config_file_path))

    try:
        config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.fastercnn.parameters.checkpoint_url)  # Let training initialize from model zoo
        config.DATASETS.TEST = (cfg.fastercnn.parameters.val_dataset_name,)
    except RuntimeError:
        config.MODEL.WEIGHTS = cfg.fastercnn.parameters.checkpoint_url
        config.DATASETS.TEST = (cfg.fastercnn.parameters.test_dataset_name,)

    config.DATASETS.TRAIN = (cfg.fastercnn.parameters.train_dataset_name,)
    config.DATALOADER.NUM_WORKERS = cfg.training.workers
    config.SOLVER.IMS_PER_BATCH = cfg.training.batch  # batch size
    config.SOLVER.BASE_LR = cfg.training.lr  # LR
    config.SOLVER.MAX_ITER = int((num_train_images / cfg.training.batch) * cfg.training.epochs)
    config.MODEL.ROI_HEADS.NUM_CLASSES = cfg.datasets.n_classes  # Set number of classes
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.test.confidence_threshold  # Set confidence score threshold for this model
    config.MODEL.ROI_HEADS.NMS_THRESH_TEST = cfg.test.iou_threshold # Set iou score threshold for this model
    config.MODEL.DEVICE = cfg.fastercnn.parameters.device  # CUDA
    config.OUTPUT_DIR = os.path.join(cfg.project_path, cfg.fastercnn.parameters.output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return config


def create_config_file(template_path, config_path, **kwargs):
    with open(template_path, "r") as template_file:
        try:
            config = template_file.read()
            config = config.format(**kwargs)
            with open(config_path, 'w') as config_file:
                config_file.write(config)
        except Exception as e:
            print(e)


@hydra.main(config_path="../../config/", config_name="config")
def prepare_config(cfg):
    '''function to create the configuration for a specific model starting from its 
    template configuration file'''

    if cfg.model == 'yolo':
        model_template_path = os.path.join(cfg.project_path, cfg.config.templates_path, cfg.yolo.yolo_templates.model_template)
        data_template_path = os.path.join(cfg.project_path, cfg.config.templates_path, cfg.yolo.yolo_templates.data_template)

        actual_config_path = os.path.join(cfg.project_path, cfg.config.actual_config_path)
        if not os.path.exists(actual_config_path):
            os.makedirs(actual_config_path)

        model_config_path = os.path.join(actual_config_path, cfg.yolo.yolo_config.model_config)
        data_config_path = os.path.join(actual_config_path, cfg.yolo.yolo_config.data_config)

        train_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.train)
        val_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.val)
        test_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.test)

        create_config_file(model_template_path, model_config_path, nc=cfg.datasets.n_classes)
        create_config_file(data_template_path, data_config_path,
                           class_list_names=cfg.datasets.class_name,
                           train_path = train_path,
                           val_path = val_path,
                           test_path = test_path
                           )
        config = get_yolo_configuration(cfg)
        return config

    if cfg.model == 'fasterRCNN':
        config = get_fastercnn_configuration(cfg)
        return config


if __name__ == '__main__':
    prepare_config()