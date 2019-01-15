import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
import numpy as np
import os
import cv2
import csv


# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from demo.predictor_rsna import RSNADemo

config_file = "/home/bong6/lib/robin_cer/configs/e2e_mask_rcnn_R_50_FPN_1x_rsna.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = RSNADemo(
    cfg,
    min_image_size=512,
    confidence_threshold=0.7,
)

def load(path):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(path).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.waitforbuttonpress()

dir_path = "/home/bong6/data/mrcnn_cer/stage1_train1/images"
anno_file = "/home/bong6/data/csv/output.csv"
result_path = '/home/bong6/data/mrcnn_cer/result_rec'

img_list = os.listdir(dir_path)

anno_dict = dict()
for filename in img_list:
    filename = os.path.splitext(filename)[0]
    anno_dict[filename] = list()

# for dirName, subdirList, fileList in os.walk(dir_path):
#     for filename in fileList:
#         filename, ext = os.path.splitext(filename)
#         if ext.lower() in [".png", ".jpg", ".jpeg"]:
#

with open(anno_file, 'r') as ann_f:
    ann_cvf = csv.reader(ann_f)

    # patientId,x,y,width,height,Target
    for i, line in enumerate(ann_cvf):
        if i == 0:
            continue

        filename, x, y, w, h, target = line
        target = int(target)

        if target == 0:
            continue
        #print(filename, x, y, w, h, target)

        x1 = int(x)
        y1 = int(y)
        w = int(w)
        h = int(h)

        x2 = x1 + w
        y2 = y1 + h

        anno_dict[filename] = (x1, y1, x2, y2, target)
        #print(anno_dict[filename])


for dirName, subdirList, fileList in os.walk(dir_path):
     for filename in fileList:

            file_path = os.path.join(dirName,filename)

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            predictions = coco_demo.run_on_opencv_image(img)

            filename = os.path.splitext(filename)[0]
            #print('filename',filename)


            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            result_dir = os.path.join(result_path)
            print('processing -> this file', filename)

            if 'Type_1' in file_path:

                result_1 = os.path.join(result_dir, "Type_1")
                if not os.path.isdir(result_1):
                    os.makedirs(result_1)
                result_1 = os.path.join(result_1, filename + '.png')

                cv2.imwrite(result_1, predictions)

            if 'Type_2' in file_path:

                result_2 = os.path.join(result_dir, "Type_2")
                if not os.path.isdir(result_2):
                    os.makedirs(result_2)
                result_2 = os.path.join(result_2, filename + '.png')
                cv2.imwrite(result_2, predictions)

            if 'Type_3' in file_path:

                result_3 = os.path.join(result_dir, "Type_3")
                if not os.path.isdir(result_3):
                    os.makedirs(result_3)
                result_3 = os.path.join(result_3, filename+'.png')
                cv2.imwrite(result_3, predictions)



            #cv2.imwrite(result_dir, predictions)
            #imshow(predictions)


