import hydra
import os
import matplotlib.pyplot as plt
import fiftyone as fo



@hydra.main(config_path="./config/", config_name="config")
def train(cfg):

    img_path = os.path.join(cfg.datasets.path, cfg.datasets.img_path)
    coco_file = os.path.join(cfg.datasets.path, cfg.preproc.preprocessed_annotation)
    
    if cfg.dataset == 'yolo':
        # TODO scrivi codice yolo
        print('ciao')
    elif cfg.dataset == 'coco':
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=img_path,
            labels_path=coco_file,
        )

        session = fo.launch_app(dataset)
        session.wait()

if __name__ == '__main__':
    view()