#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""

import pytorch_lightning as pl
import torch
from transformers import DetrForObjectDetection


class Detr(pl.LightningModule):

    def __init__(self, lr=0.0001, lr_backbone=1e-05, weight_decay=0.0001, num_labels=3, train_dataloader=None,
                 val_dataloader=None):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            num_labels=num_labels,
                                                            ignore_mismatched_sizes=True)

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.t_dataloader = train_dataloader
        self.v_dataloader = val_dataloader

    def forward(self, pixel_values, pixel_mask):

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch):

        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.common_step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):

        loss, loss_dict = self.common_step(batch)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):

        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):

        return self.t_dataloader

    def val_dataloader(self):

        return self.v_dataloader
