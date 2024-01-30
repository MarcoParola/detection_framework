# **Detection framework**


[![license](https://img.shields.io/static/v1?label=OS&message=Windows&color=green&style=plastic)]()
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=plastic)]()
[![size](https://img.shields.io/github/languages/code-size/MarcoParola/detection_framework?style=plastic)]()
[![license](https://img.shields.io/github/license/MarcoParola/detection_framework?style=plastic)]()


The project concerns the development of an object detection ensemble architecture presented at [IEEE SSCI-2023](https://attend.ieee.org/ssci-2023/). Full text is available [here](https://ieeexplore.ieee.org/document/10371865).

A python wrapping framework for performing object detection tasks using state-of-the-art deep learning architecture: YOLOv7, Faster R-CNN, DEtection TRansformer DE-TR.

<img title="a title" width="400" alt="Alt text" src="./img/ensemble-architecture.jpg">


The architecture was tested on an oral cancer dataset, below are some examples of predictions

<img title="a title" alt="Alt text" src="./img/predictions1.jpg">
<img title="a title" alt="Alt text" src="./img/predictions8.jpg">

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
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

download zip file containing images (oral1.zip) and coco dataset (oral1.json) and put cocodataset in `./data/`.



## **Usage**

### **Data preparation**
Since this framework relies on different models, different data formats are needed. 
During the project installation, 3 subfolders are created in data: orig, yolo and coco. 
The basic idea is to put your dataset-images in the orig folder; then generate your yolo/coco dataset by using some preprocessing-converter scripts. Please note: if your data doesn't required any preprocessing, you can skip this step, and directily put your data in yolo or coco folder.

```sh
sh scripts/sh/preprocessing.sh
python -m scripts.py.preprocessing.resize_image preproc.img_size.width=640 preproc.img_size.height=640
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

If you find this repo useful, please cite it as:
```
@INPROCEEDINGS{10371865,
  author={Parola, Marco and Mantia, Gaetano La and Galatolo, Federico and Cimino, Mario G.C.A. and Campisi, Giuseppina and Di Fede, Olga},
  booktitle={2023 IEEE Symposium Series on Computational Intelligence (SSCI)}, 
  title={Image-Based Screening of Oral Cancer via Deep Ensemble Architecture}, 
  year={2023},
  volume={},
  number={},
  pages={1572-1578},
  doi={10.1109/SSCI52147.2023.10371865}
}
```
