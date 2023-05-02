#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""

import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, train_json_path, test_json_path, feature_extractor, train=True):
        # ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        if train:
            ann_file = train_json_path
        else:
            ann_file = test_json_path
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target
