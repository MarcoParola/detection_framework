import json

import os
import cv2
import torch
from ultralytics import YOLO

from transformers import DetrFeatureExtractor

from models.detr.detr import Detr
from scripts.py.prepare_config import prepare_config
from models.detr.prediction import visualize_predictions, get_predictions

import numpy as np
from torchvision.io import read_image



coeff = {
    'yolo': [.457, .314, .49],
    'detr': [.631, 303, .459],
    'fasterRCNN': [.722, .279, .392]
}


class EnsembledDetector:

    def predict(self, predictions):
        for firsrt_prediction in predictions:
            for second_prediction in predictions:
                if firsrt_prediction['model'] != second_prediction['model']:
                    break

    
    def predict(self, predictions, threshold=.5):

        label_ens, bbox_ens = [],[]

        for i in range(len(predictions)):
            pred_first_model = predictions[i]
            first_model = predictions[i]['model']

            for j in range(len(predictions)):
                pred_second_model = predictions[j]
                second_model = predictions[j]['model']

                if i != j:
                    labels1, bboxes1 = pred_first_model['labels'], pred_first_model['bboxes']
                    labels2, bboxes2 = pred_second_model['labels'], pred_second_model['bboxes']
                    for label1,bbox1 in zip(labels1,bboxes1):
                        for label2,bbox2 in zip(labels2,bboxes2):
                            predicted_labels = {
                                first_model:label1, 
                                second_model:label2
                            }
                            iou = self.compute_iou(bbox1, bbox2)
                            check = False
                            if iou > threshold:
                                check = True
                                for k in range(len(predictions)): 
                                    pred_third_model = predictions[k]
                                    third_model = predictions[k]['model']

                                    if i != k and j != k:
                                        bboxes3 = pred_third_model['bboxes']
                                        labels3, bboxes3 = pred_third_model['labels'], pred_third_model['bboxes']
                                        for label3,bbox3 in zip(labels3,bboxes3):
                                            iou = self.compute_iou(bbox1, bbox3)

                                            if iou > threshold:
                                                x = max(bbox1[0],bbox2[0],bbox3[0])
                                                y = max(bbox1[1],bbox2[1],bbox3[1])
                                                w = min(bbox1[2],bbox2[2],bbox3[2])
                                                h = min(bbox1[3],bbox2[3],bbox3[3])
                                                #l = self.average_weighted_voting([label1, label2, label3])
                                                predicted_labels[third_model] = label3
                                                lbl = self.average_weighted_voting(predicted_labels)
                                                label_ens.append(lbl)
                                                bbox_ens.append([x,y,w,h])
                                                check = False
                                                #break
                            if check:
                                x = max(bbox1[0],bbox2[0])
                                y = max(bbox1[1],bbox2[1])
                                w = min(bbox1[2],bbox2[2])
                                h = min(bbox1[3],bbox2[3])
                                #l = self.average_weighted_voting([label1, label2])
                                lbl = self.average_weighted_voting(predicted_labels)
                                label_ens.append(lbl)
                                bbox_ens.append([x,y,w,h])

        bbox_ens,label_ens = self.merge_bboxes(bbox_ens, label_ens)
        return label_ens, bbox_ens

    def average_weighted_voting(self,predicted_labels):
        """
        Computes the predicted label based on the average voting

        Arguments:
        predicted_labels -- Dict {'yolo': lbl1, 'fasterRCNN': lbl2, 'detr': lbl3}. 

        Returns:
        predicted_label -- the label predicted by the ensemble model
        """
        predicted_label = -1
        preds = [0,0,0]
        for key in predicted_labels:
            preds[int(predicted_labels[key])] = coeff[key][int(predicted_labels[key])]
        predicted_label = torch.argmax(torch.Tensor(preds))
        return predicted_label

        
    def compute_iou(self,bbox1, bbox2):
        """
        Computes the Intersection over Union (IoU) metric between two bounding boxes.
        
        Arguments:
        bbox1 -- Tuple (x, y, w, h) representing the first bounding box.
        bbox2 -- Tuple (x, y, w, h) representing the second bounding box.
        
        Returns:
        iou -- The Intersection over Union (IoU) metric.
        """
        x1, y1, w1, h1 = bbox1[0],bbox1[1],bbox1[2],bbox1[3]
        x2, y2, w2, h2 = bbox2[0],bbox2[1],bbox2[2],bbox2[3]
        # Calculate the coordinates of the intersection rectangle
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        w_intersection = min(x1 + w1, x2 + w2) - x_intersection
        h_intersection = min(y1 + h1, y2 + h2) - y_intersection

        # If the intersection is non-existent (negative width or height), return IoU = 0
        if w_intersection <= 0 or h_intersection <= 0:
            return 0.0

        # Calculate the areas of the bounding boxes
        area_bbox1 = w1 * h1
        area_bbox2 = w2 * h2
        # Calculate the area of the intersection and union
        area_intersection = w_intersection * h_intersection
        area_union = area_bbox1 + area_bbox2 - area_intersection

        iou = area_intersection / area_union
        return iou


    def merge_bboxes(self, bboxes_list, votes_list):
        ''' function to generate the final (predicted) bboxes from all those detected'''

        print(len(bboxes_list), len(votes_list))
        merged_bboxes, merged_votes = [],[]
        for bbox, vote in zip(bboxes_list,votes_list):
            if len(merged_bboxes) == 0:
                merged_bboxes.append([bbox, 1])
                merged_votes.append([0,0,0])
                merged_votes[0][vote] += 1

            iou_check = True
            for i in range(len(merged_bboxes)):
                b1 = [torch.tensor(item, dtype=torch.float) for item in merged_bboxes[i][0]]
                b2 = [torch.tensor(item.clone(), dtype=torch.float) for item in bbox]
                iou = self.compute_iou(b1, b2)
                if iou > 0.5:
                    iou_check = False
                    box1 = merged_bboxes[i][0]
                    box1 = torch.tensor([box1], dtype=torch.float)
                    box2 = bbox
                    box2 = torch.tensor([box2], dtype=torch.float)
                    merged_bboxes[i][0] = ((box1*torch.tensor([merged_bboxes[i][1]], dtype=torch.float) + box2) / (merged_bboxes[i][1]+1) ).tolist()[0]
                    merged_bboxes[i][1] += 1
                    merged_votes[i][vote] += 1
            if iou_check:
                merged_bboxes.append([bbox, 1])
                merged_votes.append([0,0,0])
                merged_votes[-1][vote] += 1

        merged_bboxes = [bbox[0] for bbox in merged_bboxes]
        merged_votes = [vote.index(max(vote)) for vote in merged_votes]

        return merged_bboxes, merged_votes
