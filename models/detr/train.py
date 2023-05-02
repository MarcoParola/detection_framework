#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DetrFeatureExtractor

from .coco_detection import CocoDetection
from .datasets_helper import get_coco_api_from_dataset
from .datasets_helper.coco_eval import CocoEvaluator
from .detr import Detr

import numpy as np


class DetrTrainer:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            strict=False,
            verbose=False,
            mode='min'
        )

        self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.feature_extractor)

    @staticmethod
    def get_final_path(sub_count, join_list):

        path = os.path.dirname(os.path.realpath(__file__))
        for i in range(sub_count): path = os.path.dirname(os.path.normpath(path))
        for i in range(len(join_list)): path = os.path.join(path, join_list[i])

        return path

    @staticmethod
    def collate_fn(batch):

        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        pixel_values = [item[0] for item in batch]
        encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {'pixel_values': encoding['pixel_values'], 'pixel_mask': encoding['pixel_mask'], 'labels': labels}

        return batch

    def create_dataset(self):
        train_dataset = CocoDetection(self.image_path, self.train_json_annot_path, self.val_json_annot_path,
                                      feature_extractor=self.feature_extractor)
        val_dataset = CocoDetection(self.image_path, self.train_json_annot_path, self.val_json_annot_path,
                                    feature_extractor=self.feature_extractor, train=False)
        test_dataset = CocoDetection(self.image_path, self.train_json_annot_path, self.test_json_annot_path,
                                     feature_extractor=self.feature_extractor, train=False)

        return train_dataset, val_dataset, test_dataset

    def evaluation(self, val_dataset, val_dataloader, model):

        base_ds = get_coco_api_from_dataset(val_dataset)
        iou_types = ['bbox']
        coco_evaluator = CocoEvaluator(base_ds, iou_types)  # initialize evaluator with ground truths

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        print("Running evaluation...")

        for idx, batch in enumerate(tqdm(val_dataloader)):
            # get the inputs
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in
                      batch["labels"]]  # these are in DETR format, resized + normalized

            # forward pass
            outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
            results = self.feature_extractor.post_process(outputs,
                                                          orig_target_sizes)  # convert outputs of model to COCO api
            res = {target['image_id'].item(): output for target, output in zip(labels, results)}
            coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()

        # Initialize an array to store the AP for each class
        ap_50_per_class = np.zeros(3)
        ap_50_95_per_class = np.zeros(3)

        # Calculate class-wise AP using coco_evaluator
        for iou_type in iou_types:
            coco_eval = coco_evaluator.coco_eval[iou_type]
            for class_idx in range(3):
                ap_50_per_class[class_idx] = coco_eval.eval['precision'][0, :, class_idx, 0, -1].mean()
                ap_50_95_per_class[class_idx] = coco_eval.eval['precision'][:, :, class_idx, 0, -1].mean()

        # Print the mAP for each class
        for class_idx, ap in enumerate(ap_50_per_class):
            print(f"mAP_50 for class {class_idx}: {ap:.4f}")

        for class_idx, ap in enumerate(ap_50_95_per_class):
            print(f"mAP_50:95 for class {class_idx}: {ap:.4f}")

        coco_evaluator.summarize()

    def data_loader(self, dataset, batch_size, shuffle=False):
        dataloader = DataLoader(dataset, collate_fn=DetrTrainer.collate_fn, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def build_model(self, train_dataloader, val_dataloader):
        model = Detr(lr=self.lr, lr_backbone=self.lr_backbone, weight_decay=self.weight_decay,
                     num_labels=self.num_classes, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
        return model

    def train(self, train_dataset, val_dataset):
        train_dataloader = self.data_loader(train_dataset, self.train_batch_size, shuffle=True)
        val_dataloader = self.data_loader(val_dataset, self.test_batch_size)

        model = Detr(lr=self.lr, lr_backbone=self.lr_backbone, weight_decay=self.weight_decay,
                     num_labels=self.num_classes, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

        # Set custom logger with desired output directory
        logs_path = self.output_path
        logger = TensorBoardLogger(save_dir=logs_path, name=self.logs_dir)

        #PATH = 'C:/Users/fuma2/Development/Github/detection_framework/outputs/detr/model.pth'
        #model = model.load_from_checkpoint(PATH,lr=self.lr, lr_backbone=self.lr_backbone, weight_decay=self.weight_decay,
        #             num_labels=self.num_classes, train_dataloader=train_dataloader, val_dataloader=val_dataloader)

        trainer = Trainer(max_epochs=self.max_epochs, gradient_clip_val=self.gradient_clip_val, logger=logger,
                          callbacks=[self.early_stop])
        trainer.fit(model)

        # -----
        self.evaluation(val_dataset, val_dataloader, model)

        return model, trainer

    def main(self):
        train_dataset, val_dataset, test_dataset = self.create_dataset()
        _, trainer = self.train(train_dataset, val_dataset)

        logs_dir = os.path.join(self.output_path, self.logs_dir)
        # find the last run's version number by looking at the subdirectories of logs_dir
        version_nums = [int(dir_name.split("_")[-1]) for dir_name in os.listdir(logs_dir) if
                        dir_name.startswith("version_")]
        last_version_num = max(version_nums) if version_nums else 0

        version_dir = os.path.join(logs_dir, f"version_{last_version_num}")
        # specify the path where the model.pth file will be saved
        model_path = os.path.join(version_dir, self.model_path)

        trainer.save_checkpoint(model_path)

        return
