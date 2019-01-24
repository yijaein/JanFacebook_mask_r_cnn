# robin_cer readme



## Nuclei Counting and Segmentation

This sample implements the [2018 Data Science Bowl challenge](https://www.kaggle.com/c/data-science-bowl-2018).
The goal is to segment individual nuclei in microscopy images.
The `nucleus.py` file contains the main parts of the code, and the two Jupyter notebooks


## Command line Usage
Train a new model starting from ImageNet weights using `train` dataset (which is `stage1_train` minus validation set)

    python3 nucleus.py train --dataset=/home/bong6/data/mrcnn_cer --subset=train --weights=imagenet
    python3 nucleus.py detect --dataset=/home/bong6/data/mrcnn_cer --subset=stage1_test --weights=/home/bong6/lib/robin_cer/logs/nucleus20181218T1445/mask_rcnn_nucleus_0025.h5

Train a new model starting from specific weights file using the full `stage1_train` dataset

    python3 nucleus.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5

Resume training a model that you had trained earlier

    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last


Generate submission file from `stage1_test` images

    python3 nucleus.py detect --dataset=/path/to/dataset --subset=stage1_test --weights=<last or /path/to/weights.h5>