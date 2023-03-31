import hydra
import os



@hydra.main(config_path="../../config/", config_name="config")
def prepare_config(cfg):
    '''function to create the configuration for a specific model starting from its 
    template configuration file'''

    if cfg.model == 'yolo':

        # create configuration file for model
        with open("../../../config/yolov8-model-template.yaml", "r") as template_file:
            try:
                # insert actual config value parameters in the template configuration file
                yolo_config = template_file.read()
                yolo_config = yolo_config.format(nc=cfg.datasets.n_classes)
                with open('../../../config/yolov8-model.yaml', 'w') as config_file:
                    config_file.write(yolo_config)
            except Exception as e:
                print(e)

        # create configuration file for data
        with open("../../../config/yolov8-data-template.yaml", "r") as template_file:
            try:
                # insert actual config value parameters in the template configuration file
                yolo_config = template_file.read()
                yolo_config = yolo_config.format(class_list_names=cfg.datasets.class_name)
                with open('../../../config/yolov8-data.yaml', 'w') as config_file:
                    config_file.write(yolo_config)
            except Exception as e:
                print(e)

    if cfg.model = 'fasterRCNN':
        #...
        print('coco')


if __name__ == '__main__':
    prepare_config()