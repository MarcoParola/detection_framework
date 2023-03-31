import hydra
import os
from ultralytics import YOLO
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo



def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, val_dataset_name, num_classes, device,
                  output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)  # Let training initialize from model zoo
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.SOLVER.IMS_PER_BATCH = 4  # batch size
    cfg.SOLVER.BASE_LR = 0.001  # LR
    cfg.SOLVER.MAX_ITER = 10000  # longer for difficult dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Set number of classes
    cfg.MODEL.DEVICE = device  # CUDA
    cfg.OUTPUT_DIR = output_dir

    return cfg

@hydra.main(config_path="./config/", config_name="config")
def train(cfg):
    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, 'config', 'yolov8.yaml')
        data_path = os.path.join(cfg.project_path, 'config', 'yolodata.yaml')

        model = YOLO(model_path).load('yolov8n.pt')  # build from YAML and transfer weights
        # Train the model
        model.train(data=data_path, epochs=50, imgsz=640, workers=8, device=0)

    if cfg.model == 'coco':
        config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

        output_dir = os.path.join(cfg.project_path,"outputs/coco_object_detection")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        num_classes = 3

        device = "cuda"

        train_dataset_name = "oralcancer_train"
        val_dataset_name = "oralcancer_val"
        test_dataset_name = "oralcancer_test"

        images_path = os.path.join(cfg.datasets.path, "coco", "aug_images")

        train_json_annot_path = os.path.join(cfg.datasets.path, "coco", "train.json")
        val_json_annot_path = os.path.join(cfg.datasets.path, "coco", "val.json")
        test_json_annot_path = os.path.join(cfg.datasets.path, "coco", "test.json")

        # Register the dataset for the model usages
        register_coco_instances(train_dataset_name, {}, train_json_annot_path, images_path)
        register_coco_instances(val_dataset_name, {}, val_json_annot_path, images_path)
        register_coco_instances(test_dataset_name, {}, test_json_annot_path, images_path)

        cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, val_dataset_name, num_classes, device, output_dir)

        trainer = DefaultTrainer(cfg)
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