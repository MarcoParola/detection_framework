#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: A.Akdogan
"""

import os

import matplotlib.pyplot as plt
import torch

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def get_max_prob(prob):
    probability = 0
    for p in prob:
        if p[p.argmax()] > probability:
            probability = p[p.argmax()]
    return probability

def plot_results(pil_img, prob, boxes, output_folder, image_name, classes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    max_prob = get_max_prob(prob)


    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        if (p[p.argmax()] == max_prob):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{classes[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    plt.savefig(os.path.join(output_folder, image_name))

    print("Saved the image: " + image_name)

    # Close the figure to avoid the "More than 20 figures have been opened" warning
    plt.close()


def visualize_predictions(image, outputs, output_folder, image_name, classes, threshold=0.1):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

    # plot results
    plot_results(image, probas[keep], bboxes_scaled, output_folder, image_name, classes)
