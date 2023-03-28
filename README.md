# **Detection framework**

[![license](https://img.shields.io/static/v1?label=OS&message=linux|osx&color=green&style=plastic)]()
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=plastic)]()
[![size](https://img.shields.io/github/languages/code-size/MarcoParola/detection_framework?style=plastic)]()
[![license](https://img.shields.io/github/license/MarcoParola/detection_framework?style=plastic)]()




A python wrapping framework for performing object detection tasks using deep learning architecture:
- YOLOv7
- Faster R-CNN
- DEtection TRansformer DE-TR




## **Installation**

To install the framework, simply clone the repository and install the necessary dependencies:
```sh
git clone https://github.com/MarcoParola/detection_framework.git
cd detection_framework
mkdir models data data/orig data/yolo data/coco

#TODO quando scarichi il file json, rinominalo con "coco_dataset.json"
```

create and activate virtual environment, then install dependencies. 
```sh
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt 
```

download zip file containing images (oral1.zip) and coco dataset (oral1.json) and put cocodataset in `./data/`.



## **Usage**

### **Data preparation**
Since this framework relies on different models, different data formats are needed. 
During the project installation, 3 subfolders are created in data: orig, yolo and coco. 
The basic idea is to put your dataset-images in the orig folder; then generate your yolo/coco dataset by using some preprocessing-converter scripts. Please note: if your data doesn't required any preprocessing, you can skip this step, and directily put your data in yolo or coco folder.

```sh
sh scripts/sh/clean.sh
python -m scripts.py.preprocessing.resize_images preproc.img_size.width=640 preproc.img_size.height=640
```



### **fine-tune a model**
The basic command to find-tune a model is the following

> python train.py model=*model_name* dataset=*dataset_type* 

Where ``model`` can assume the following value: 
* yolo
* fasterRCNN
* detr

while ``dataset`` can assume "coco" or "yolo"


The default folder for the images is ``./data/images/``, if you want put your file in a different folder, override the ``datasets.img_path`` argument:

> python train.py model=fasterRCNN dataset=coco.json datasets.img_path=**new_img_path**

To specify the name with which to save the model after fine tuning you can use the ``model_name`` argument:

> python train.py model=fasterRCNN dataset=coco.json model_name=**name**



