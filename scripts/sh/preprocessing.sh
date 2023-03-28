# wget link oral1.zip 
# wget link dataset.json


unzip ./data/orig/oral1.zip -d ./data/orig/tmp/
python scripts/py/preprocessing/clean_data.py 
python scripts/py/preprocessing/resize_image.py preproc.img_size.width=640 preproc.img_size.height=640

