import json
import math

import hydra
import os

from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.data.datasets import register_coco_instances


def get_yolo_configuration(cfg, mode):
    """ Obtain the yolo configuration to be used for train or test """
    if (mode == "train"):
        data_path = os.path.join(cfg.project_path, cfg.config.actual_config_path,
                                 cfg.yolo.yolo_config.data_config_train)
    else:
        data_path = os.path.join(cfg.project_path, cfg.config.actual_config_path,
                                 cfg.yolo.yolo_config.data_config_test)

    config = {
        "project": os.path.join(cfg.project_path, cfg.yolo.parameters.output_dir),
        "data": data_path,
        "lr0": cfg.training.lr,
        "epochs": cfg.training.epochs,
        "batch": cfg.training.batch,
        "patience": cfg.training.early_stopping.patience,
        "optimizer": cfg.training.optimizer,
        "device": cfg.yolo.parameters.device,
        "workers": cfg.training.workers,
        "imgsz": cfg.training.img_size
    }

    return config


def get_detr_configuration(cfg):
    """ Obtain the detr configuration to be used for train or test """
    output_path = os.path.join(cfg.project_path, cfg.detr.parameters.output_dir)

    config = {
        "image_path": os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path),
        "train_json_annot_path": os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.train),
        "val_json_annot_path": os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.val),
        "test_json_annot_path": os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.test),
        "output_path": output_path,
        "model_path": cfg.detr.detr_model_path,

        "feature_extractor": cfg.detr.parameters.feature_extractor,
        "train_batch_size": cfg.training.batch,
        "test_batch_size": cfg.training.val_batch,
        "lr": cfg.training.lr,
        "lr_backbone": cfg.detr.parameters.lr_backbone,
        "weight_decay": cfg.training.weight_decay,
        "max_epochs": cfg.training.epochs,
        "gradient_clip_val": cfg.detr.parameters.gradient_clip_val,
        "patience": cfg.training.early_stopping.patience,

        "num_classes": cfg.datasets.n_classes,

        "logs_dir": cfg.detr.parameters.logs_dir
    }

    return config


def get_num_images(json_path):
    with open(json_path, "r") as f:
        dataset = json.load(f)
    image_ids = [image['id'] for image in dataset['images']]
    return len(image_ids)


def get_fastercnn_configuration(cfg, mode):
    """ Obtain the fastercnn configuration to be used for train or test """

    images_path = os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path)
    output_dir = os.path.join(cfg.project_path, cfg.fastercnn.parameters.output_dir)

    train_json_annot_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.train)
    val_json_annot_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.val)
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

    # Create configuration
    config = get_cfg()

    config.merge_from_file(model_zoo.get_config_file(cfg.fastercnn.parameters.config_file_path))
    if mode == "train":
        config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            cfg.fastercnn.parameters.checkpoint_url)  # Let training initialize from model zoo
        config.DATASETS.TEST = (cfg.fastercnn.parameters.val_dataset_name,)  # Use the val dataset
    else:
        config.MODEL.WEIGHTS = os.path.join(output_dir,
                                            cfg.fastercnn.fastercnn_model_path)  # Use the trained model for the test
        config.DATASETS.TEST = (cfg.fastercnn.parameters.test_dataset_name,)  # Use the test dataset

    config.DATASETS.TRAIN = (cfg.fastercnn.parameters.train_dataset_name,)

    config.DATALOADER.NUM_WORKERS = cfg.training.workers

    config.SOLVER.IMS_PER_BATCH = cfg.training.batch  # batch size
    config.SOLVER.BASE_LR = cfg.training.lr  # LR
    config.SOLVER.MAX_ITER = math.ceil(
        num_train_images / cfg.training.batch * cfg.training.epochs)  # Compute max_iter to get the right amount of epochs

    config.MODEL.ROI_HEADS.NUM_CLASSES = cfg.datasets.n_classes  # Set number of classes
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.test.confidence_threshold  # Set confidence score threshold for this model
    config.MODEL.ROI_HEADS.NMS_THRESH_TEST = cfg.test.iou_threshold  # Set iou score threshold for this model
    config.MODEL.DEVICE = cfg.fastercnn.parameters.device  # CUDA

    config.TEST.EVAL_PERIOD = math.ceil(
        num_train_images / cfg.training.batch)  # Eval the quality of the models at each epoch

    config.OUTPUT_DIR = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return config


def create_config_file(template_path, config_path, **kwargs):
    """function to create a configuration file given a template"""
    with open(template_path, "r") as template_file:
        try:
            config = template_file.read()
            config = config.format(**kwargs)
            with open(config_path, 'w') as config_file:
                config_file.write(config)
        except Exception as e:
            print(e)


def prepare_config(cfg, mode):
    """function that returns the configuration of each model to be used for training or test"""

    if cfg.model == 'yolo':
        model_template_path = os.path.join(cfg.project_path, cfg.config.templates_path,
                                           cfg.yolo.yolo_templates.model_template)
        data_template_path = os.path.join(cfg.project_path, cfg.config.templates_path,
                                          cfg.yolo.yolo_templates.data_template)

        actual_config_path = os.path.join(cfg.project_path, cfg.config.actual_config_path)
        if not os.path.exists(actual_config_path):
            os.makedirs(actual_config_path)

        model_config_path = os.path.join(actual_config_path, cfg.yolo.yolo_config.model_config)
        data_config_path = os.path.join(actual_config_path, cfg.yolo.yolo_config.data_config_train)
        data_config_path_test = os.path.join(actual_config_path, cfg.yolo.yolo_config.data_config_test)

        train_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.train)
        val_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.val)
        test_path = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.test)

        # Create actual_config yaml file from the templates
        create_config_file(model_template_path, model_config_path, nc=cfg.datasets.n_classes)
        create_config_file(data_template_path, data_config_path,
                           class_list_names=cfg.datasets.class_name,
                           train_path=train_path,
                           val_path=val_path
                           )

        create_config_file(data_template_path, data_config_path_test,
                           class_list_names=cfg.datasets.class_name,
                           train_path=train_path,
                           val_path=test_path
                           )

        config = get_yolo_configuration(cfg, mode)

        return config

    if cfg.model == 'fasterRCNN':
        config = get_fastercnn_configuration(cfg, mode)

        return config

    if cfg.model == 'detr':
        config = get_detr_configuration(cfg)

        return config


@hydra.main(config_path="../../config/", config_name="config", version_base=None)
def main(cfg):
    prepare_config(cfg, mode="train")


if __name__ == '__main__':
    main()
