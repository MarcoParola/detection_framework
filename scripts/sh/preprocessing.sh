wget https://drive.google.com/file/d/1xW63RZTvxrnLzTzpUx0kkh8d9IWepN4_/view?usp=sharing -P data/orig
wget https://drive.google.com/file/d/1deqYC1PmjpMYDQP4DrELxTr25MFGGnzo/view?usp=share_link -P data/orig


unzip ./data/orig/oral1.zip -d ./data/orig/tmp/
python scripts/py/preprocessing/clean_data.py 
python scripts/py/preprocessing/resize_image.py preproc.img_size.width=640 preproc.img_size.height=640

mkdir data/coco
mkdir data/yolo

