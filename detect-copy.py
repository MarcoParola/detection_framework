import json

import hydra
import os
import cv2
import torch

from ultralytics import YOLO

from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from transformers import DetrFeatureExtractor

from models.detr.detr import Detr
from models.ensemble.detector import EnsembledDetector
from scripts.py.prepare_config import prepare_config
from models.detr.prediction import visualize_predictions, get_predictions

from PIL import Image, ImageDraw

import numpy as np
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint

import tensorflow as tf
from tensorflow import keras
import keras_cv


font = cv2.FONT_HERSHEY_SIMPLEX
border_size = 4

def plot_rect_and_text(img, bbox, text):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=border_size)
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+237, int(bbox[1])-50), color=(0, 255, 0), thickness=-1)
    cv2.putText(img, text, (int(bbox[0]+4), int(bbox[1])-10), fontScale=1.46, fontFace=font, color=(0, 0, 0), thickness=border_size)



def compute_iou(bbox1, bbox2):
    # Extract coordinates from the bounding boxes
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Calculate the intersection area
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calculate the union area
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou



def compute_metric_map(actual, predicted):
    num_images = len(actual)
    average_precisions = []

    map50_95 = []
    
    for map_step in np.arange(.5,1.,.05):
        for i in range(num_images):
            image_actual = actual[i]
            image_predicted = predicted[i]
            
            sorted_predicted = image_predicted
            sorted_actual = image_actual
            
            num_predictions = len(sorted_predicted)
            true_positives = np.zeros(num_predictions)
            false_positives = np.zeros(num_predictions)
            precision = []
            recall = []
            
            num_actual = len(sorted_actual)
            is_true_positive = np.zeros(num_actual, dtype=bool)
            
            for j, pred in enumerate(sorted_predicted):
                best_iou = 0.0
                best_match = -1
                
                for k, actual_bbox in enumerate(sorted_actual):
                    iou = compute_iou(pred[:-2], actual_bbox[:-1])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = k
                
                if best_iou >= map_step and not is_true_positive[best_match] and pred[-2]==actual_bbox[-1]:
                    true_positives[j] = 1
                    is_true_positive[best_match] = True
                else:
                    false_positives[j] = 1
                
                precision.append(np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives)))
                recall.append(np.sum(true_positives) / num_actual)
            
            average_precision = 0.0
            previous_recall = 0.0
            for prec, rec in zip(precision, recall):
                if np.isnan(prec):
                    prec = 0
                if np.isnan(rec):
                    rec = 0
                average_precision += (rec - previous_recall) * prec
                previous_recall = rec
            
            mean_average_precision = np.mean(average_precision)
            average_precisions.append(mean_average_precision)
        
        mean_average_precision_dataset = np.mean(average_precisions)
        map50_95.append(mean_average_precision_dataset)

    return np.mean(map50_95)
    #return mean_average_precision_dataset



@hydra.main(config_path="./config/", config_name="config", version_base=None)
def detect(cfg):
    if cfg.model == 'yolo':
        model_path = os.path.join(cfg.project_path, cfg.yolo.parameters.output_dir, cfg.yolo.yolo_model_path)
        model = YOLO(model_path)  # load a custom model

        # define paths to input and output folders
        input_folder = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.test, cfg.datasets.img_path)
        output_folder = os.path.join(cfg.project_path, cfg.yolo.yolo_detect_output_path)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # loop over each image in the input folder
        for image_name in os.listdir(input_folder):
            # read image
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)
            # detect objects and get bounding boxes
            res = model(image)
            res_plotted = res[0].plot()
            # save image with bounding boxes
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, res_plotted)


    if cfg.model == 'fasterRCNN':
        output_folder = os.path.join(cfg.project_path, cfg.fastercnn.fastercnn_detect_output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cfg.fastercnn.parameters.checkpoint_url = os.path.join(cfg.project_path, cfg.fastercnn.parameters.output_dir,
                                                               cfg.fastercnn.fastercnn_model_path)
        config = prepare_config(cfg, 'test')
        predictor = DefaultPredictor(config)
        test_dataset_dicts = DatasetCatalog.get(cfg.fastercnn.parameters.test_dataset_name)
        # Loop over each image in the test dataset
        for d in test_dataset_dicts:
            # Load the image
            img = cv2.imread(d["file_name"])
            # Use the predictor to generate predictions for the image
            outputs = predictor(img)
            # Get the predicted instances with the highest confidence scores
            instances = outputs["instances"]
            scores = instances.scores.tolist()
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]
            instances = instances[indices]
            # Visualize the predictions on the image
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.fastercnn.parameters.test_dataset_name))
            v = v.draw_instance_predictions(instances.to("cpu"))
            # Save the image with the bounding boxes
            output_path = os.path.join(output_folder, os.path.basename(d["file_name"]))
            cv2.imwrite(output_path, v.get_image()[:, :, ::-1])


    if cfg.model == "detr":
        # define paths to input and output folders
        input_folder = os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path)
        output_folder = os.path.join(cfg.project_path, cfg.detr.detr_detect_output_path)
        test_annotation_file = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.test)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(test_annotation_file, 'r') as f:
            test_data = json.load(f)

        # Define the model and the feature extractor
        model_path = os.path.join(os.path.join(cfg.project_path, cfg.detr.parameters.output_dir),
                                  cfg.detr.detr_model_path)
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        model = Detr(num_labels=cfg.datasets.n_classes)
        model = model.load_from_checkpoint(model_path)
        model.eval()

        # Apply detection to each test image
        for image_info in test_data["images"]:
            image_name = image_info["file_name"]
            image_path = os.path.join(input_folder, image_name)
            img = Image.open(image_path)
            encoding = feature_extractor(img, return_tensors="pt")
            encoding.keys()
            outputs = model(**encoding)
            visualize_predictions(img, outputs, output_folder, image_name, cfg.datasets.class_name)


    preds, targets = [],[]

    if cfg.model == "ensemble":
        
        # YOLO
        model_path = os.path.join(cfg.project_path, cfg.yolo.parameters.output_dir, cfg.yolo.yolo_model_path)
        model_yolo = YOLO(model_path)  # load a custom model
        input_folder_yolo = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.yolo.test, cfg.datasets.img_path)
        test_yolo = os.listdir(input_folder_yolo)

        # FASTER-RCNN
        cfg.fastercnn.parameters.checkpoint_url = os.path.join(cfg.project_path, cfg.fastercnn.parameters.output_dir,
                                                               cfg.fastercnn.fastercnn_model_path)
        cfg.model='fasterRCNN'
        config = prepare_config(cfg, 'test')
        model_fasterRCNN = DefaultPredictor(config)
        test_dataset_dicts = DatasetCatalog.get(cfg.fastercnn.parameters.test_dataset_name)

        # DETR
        input_folder_detr = os.path.join(cfg.project_path, cfg.preproc.augmentation.img_path)
        test_annotation_file = os.path.join(cfg.datasets.path, cfg.datasets.datasets_path.coco.test)
        with open(test_annotation_file, 'r') as f:
            test_data = json.load(f)
        # Define the model and the feature extractor
        model_path = os.path.join(os.path.join(cfg.project_path, cfg.detr.parameters.output_dir),
                                  cfg.detr.detr_model_path)
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        model_detr = Detr(num_labels=cfg.datasets.n_classes)
        model_detr = model_detr.load_from_checkpoint(model_path)
        model_detr.eval()

        

        # prepare actual values to compute metric
        for i in range(len(test_data['images'])):
            target = []
            d = test_dataset_dicts[i]
            for annotation in d['annotations']:
                bb = annotation['bbox']
                target.append(np.array([bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3],annotation['category_id']]))
            targets.append(np.array(target))
        targets = np.array(targets)
        #targets = targets.astype(np.float32)
           
        
        for i in range(len(test_yolo)):
            # YOLO
            image_name = test_yolo[i]
            image_path = os.path.join(input_folder_yolo, image_name)
            image_yolo = cv2.imread(image_path)
            image_multiple_bboxes = cv2.imread(image_path)
            # detect objects and get bounding boxes
            res = model_yolo(image_yolo)
            bbox_yolo, label_yolo = res[0].boxes.boxes, res[0].boxes.cls
            for bbox, label in zip( bbox_yolo, label_yolo ):
                plot_rect_and_text(image_yolo, bbox, cfg.datasets.class_name[int(label.item())])
                cv2.rectangle(image_multiple_bboxes, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=border_size)
            

            # FASTER-RCNN
            d = test_dataset_dicts[i]
            img_fasterRCNN = cv2.imread(d["file_name"])
            # Use the predictor to generate predictions for the image
            outputs = model_fasterRCNN(img_fasterRCNN)
            # Get the predicted instances with the highest confidence scores
            instances = outputs["instances"]
            scores = instances.scores.tolist()
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:1]
            instances = instances[indices]
            bbox_fasterRCNN, label_fasterRCNN = instances.pred_boxes, instances.pred_classes
            for bbox, label in zip( bbox_fasterRCNN, label_fasterRCNN ):
                plot_rect_and_text(img_fasterRCNN, bbox, cfg.datasets.class_name[int(label.item())])
                cv2.rectangle(image_multiple_bboxes, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255, 0, 0), thickness=border_size)
            

            # DETR
            image_name = d["file_name"]
            image_path = os.path.join(input_folder_detr, image_name)
            img_detr = Image.open(image_path)
            encoding = feature_extractor(img_detr, return_tensors="pt")
            encoding.keys()
            outputs = model_detr(**encoding)
            probas = outputs.logits.softmax(-1)[0, :, :-1]
            threshold=0.1
            keep = probas.max(-1).values > threshold
            label_detr, bbox_detr = get_predictions(img_detr, outputs, '', image_name, cfg.datasets.class_name)
            label_detr = torch.argmax(label_detr, dim=1)
            print('LABEL DETR:', label_detr)
            img_detr = np.asarray(img_detr)
            for bbox, label in zip( bbox_detr, label_detr):
                plot_rect_and_text(img_detr, bbox, cfg.datasets.class_name[int(label.item())])
                cv2.rectangle(image_multiple_bboxes, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=border_size)
            

            predictions = [
                {
                    'model' : 'yolo',
                    'labels': label_yolo,
                    'bboxes': bbox_yolo }, 
                {
                    'model' : 'fasterRCNN',
                    'labels': label_fasterRCNN,
                    'bboxes': bbox_fasterRCNN }, 
                {
                    'model' : 'detr',
                    'labels': label_detr,
                    'bboxes': bbox_detr }, 
            ]



            # ENSEMBLE
            img_ens = Image.open(image_path)
            img_ens = np.asarray(img_ens)
            ens_detector = EnsembledDetector()
            label_ens, bbox_ens = ens_detector.predict(predictions, .45)
            for bbox, label in zip( bbox_ens, label_ens ):
                print(bbox, label)
                plot_rect_and_text(img_ens, bbox, cfg.datasets.class_name[label])
                cv2.rectangle(image_multiple_bboxes, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 0), thickness=3)
            

            # GROUND THRUTH 
            
            img = Image.open(image_path)
            
            for annotation in d['annotations']:
                img = np.asarray(img)
                bbox = annotation['bbox']
                bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3]) ]
                label = annotation['category_id']
                segmentation = annotation['segmentation'][0]
                plot_rect_and_text(img, bbox, cfg.datasets.class_name[label])
                cv2.rectangle(image_multiple_bboxes, (bbox[0], bbox[1]), ( bbox[2], bbox[3] ), color=(255, 255, 255), thickness=3)
            

            '''
            pred = {
                'boxes':[],
                'scores':[],
                'labels':[]
            }
            for bbox, label in zip(bbox_ens,label_ens):
                pred['boxes'].append(bbox)
                pred['scores'].append(1.)
                pred['labels'].append(label)
            
            pred['boxes'] = torch.Tensor(pred['boxes']) 
            pred['scores'] = torch.Tensor(pred['scores']) 
            pred['labels'] = torch.Tensor(pred['labels']) 
            preds.append(pred)
            '''


            pred = []
            for bbox, label in zip(bbox_ens,label_ens):
                pred.append(np.array(bbox+[label, 1.]))
            preds.append(np.array(pred))
            

            plt.rcParams.update({'font.size': 13})
            # PLOT VARI
            img = np.asarray(img)
            plt.figure(figsize=(18,3.3))
            plt.subplots_adjust(left=0.01, bottom=0.001, right=0.99, top=.999, wspace=0.1, hspace=0.01)
            plt.subplot(151)
            plt.imshow(img)
            plt.title('Ground truth') 
            plt.xticks([], [])
            plt.yticks([], [])

            plt.subplot(152)
            plt.imshow(image_yolo[...,::-1])
            plt.title('YOLOv8') 
            plt.xticks([], [])
            plt.yticks([], [])

            plt.subplot(153)
            plt.imshow( img_fasterRCNN[...,::-1])
            plt.title('FasterRCNN') 
            plt.xticks([], [])
            plt.yticks([], [])

            plt.subplot(154)
            plt.imshow(img_detr)
            plt.title('DETR') 
            plt.xticks([], [])
            plt.yticks([], [])

            plt.subplot(155)
            plt.imshow(img_ens)
            plt.title('Ensemble') 
            plt.xticks([], [])
            plt.yticks([], [])
            
            plt.show()
            '''
            plt.imshow(image_multiple_bboxes[...,::-1])
            plt.title('test') 
            plt.xticks([], [])
            plt.yticks([], [])
            plt.show()
            '''

        preds = np.array(preds)
        np.save('preds.npy', preds)
        preds = np.load('preds.npy', allow_pickle=True)

        mAP = compute_metric_map(targets, preds)
        print("mAP:", mAP)
         


if __name__ == '__main__':
    detect()
    #main()