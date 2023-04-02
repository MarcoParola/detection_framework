from collections import defaultdict
import cv2
import json
import hydra
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox

def build_dictionaries(data):
    print("Building dictionaries...")
    anns = defaultdict(list)
    annsIdx = dict()
    for i in range(0, len(data['annotations'])):
        anns[data['annotations'][i]['image_id']].append(data['annotations'][i])
        annsIdx[data['annotations'][i]['id']] = i
        print("Dictionnaries built.")
    return anns, annsIdx

def resizeImageAndBoundingBoxes(imgFile, bboxes, inputW, inputH, targetImgW, targetImgH, outputImgFile):
    print("Reading image {0} ...".format(imgFile))
    img = cv2.imread(imgFile)

    seq = iaa.Sequential([
        iaa.CropToSquare(position="center"),  # crop the image to a square shape with the center of the original image as the center of the cropped image
        iaa.Resize({"height": targetImgH, "width": targetImgW}),  # resize the cropped image to the target size of (targetImgW, targetImgH)
        iaa.PadToFixedSize(width=targetImgW, height=targetImgH)  # add padding to the image to make sure it has dimensions of (targetImgW, targetImgH)
    ])
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bboxes)

    print("Writing resized image {0} ...".format(outputImgFile))
    cv2.imwrite(outputImgFile, image_aug)
    print("Resized image {0} written successfully.".format(outputImgFile))

    return bbs_aug



@hydra.main(config_path="../../../config/", config_name="config")
def resize(cfg):
    imageDir = os.path.join(cfg.project_path, cfg.preproc.orig.img_path)
    annotationsFile = os.path.join(cfg.datasets.path, cfg.datasets.original_data, 'preprocessed_' + cfg.datasets.filenames.dataset)
    targetImgW = cfg.preproc.img_size.width
    targetImgH = cfg.preproc.img_size.height
    outputImageDir = os.path.join(cfg.datasets.path, cfg.datasets.img_path)
    outputAnnotationsFile = os.path.join(cfg.datasets.path, cfg.preproc.preprocessed_annotation)

    if not os.path.exists(outputImageDir):
        os.makedirs(outputImageDir)

    print("Loading annotations file...")
    data = json.load(open(annotationsFile, 'r'))
    print("Annotations file loaded.")

    annotations, annotationsIdx = build_dictionaries(data)


    for img in data['images']:
        print("Processing image file {0} and its bounding boxes...".format(img['file_name']))

        annList = annotations[img['id']]

        # Convert COCO format bounding boxes to imgaug format
        bboxesList = []
        for ann in annList:
            bboxData = ann['bbox']
            bboxesList.append(
                BoundingBox(x1=bboxData[0], y1=bboxData[1], x2=bboxData[0] + bboxData[2], y2=bboxData[1] + bboxData[3]))

        imgFullPath = os.path.join(imageDir, img['file_name'])
        outputImgFullPath = os.path.join(outputImageDir, img['file_name'])

        outNewBBoxes = resizeImageAndBoundingBoxes(imgFullPath, bboxesList, int(img['width']), int(img['height']),
                                                   targetImgW, targetImgH, outputImgFullPath)

        for i in range(0, len(annList)):
            annId = annList[i]['id']

            x1_clipped = max(0,outNewBBoxes[i].x1)
            y1_clipped = max(0,outNewBBoxes[i].y1)
            x2_clipped = min(targetImgW,outNewBBoxes[i].x2)
            y2_clipped = min(targetImgH,outNewBBoxes[i].y2)
            width_clipped = x2_clipped - x1_clipped
            height_clipped = y2_clipped - y1_clipped

            data['annotations'][annotationsIdx[annId]]['bbox'][0] = round(float(x1_clipped),1)
            data['annotations'][annotationsIdx[annId]]['bbox'][1] = round(float(y1_clipped),1)
            data['annotations'][annotationsIdx[annId]]['bbox'][2] = round(float(width_clipped),1)
            data['annotations'][annotationsIdx[annId]]['bbox'][3] = round(float(height_clipped),1)

        img['width'] = targetImgW
        img['height'] = targetImgH

    print("Writing modified annotations to file...")
    with open(outputAnnotationsFile, 'w') as outfile:
        json.dump(data, outfile)

    print("Finished.")

    return


if __name__ == '__main__':
    resize()