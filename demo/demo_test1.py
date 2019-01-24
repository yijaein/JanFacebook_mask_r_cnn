import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from PIL import Image
import numpy as np
import os
import cv2
import csv
import PIL.Image as image


# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from demo.predictor_rsna import RSNADemo


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def classify_labels(top_prediction_label, cls, correct_label, wrong_label):

    if top_prediction_label == cls:
        correct_label +=1
    else:
        wrong_label +=1

    return correct_label,wrong_label


def extract_top_prediction(xyxy):
    asdfasd = xyxy.numpy()
    xyxyresult = []
    for item in asdfasd:
        xyxyresult.extend(item)
    # print('asfd', xyxyresult)


    if len(xyxyresult) > 4:
        xyxyresult = xyxyresult[:4]
        # print(xyxyresult)

    return xyxyresult


def crop_and_save_img(xyxy):
    x, y, w, h = xyxyresult

    result_image = img[int(y):int(y + h), int(x):int(x + w)]
    cv2.imwrite(crop_path, result_image)

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


dir_path = "/home/bong6/data/mrcnn_cer/stage1_train1/val2/images"
anno_file = "/home/bong6/data/csv/output.csv"
result_path = '/home/bong6/data/mrcnn_cer/result_rec'
crop_dir = '/home/bong6/data/mrcnn_cer/crop_image'
img_list = os.listdir(dir_path)

anno_dict = dict()
for filename in img_list:
    filename = os.path.splitext(filename)[0]
    anno_dict[filename] = list()

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

        x1, y1 = int(x), int(y)
        w, h = int(w), int(h)

        x2 = x1 + w
        y2 = y1 + h

        anno_dict[filename] = (x1, y1, x2, y2, target)
        # print(anno_dict[filename])



count = 0
iou_sum = 0
iou_average =0
correct_label = 0
wrong_label = 0
wholedataset = 300

for dirName, subdirList, fileList in os.walk(dir_path):
     for filename in fileList:

            crop_path = os.path.join(crop_dir, filename+'.png')
            file_path = os.path.join(dirName,filename)
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            predictions, x = coco_demo.run_on_opencv_image(img)

            bbox = x['bbox']
            label = x['labels']
            prediction = x['pred']

            #top prediction label extract
            top_prediction = x['top_pred']
            top_prediction_label = x['top_label']
            top_prediction_label = top_prediction_label.numpy()

            if top_prediction_label.size == 0:
                count = count+1
                print('count',count)
                continue


            top_prediction_label = top_prediction_label[0]
            label = label.numpy()

            #ex)155c no ext
            filename = os.path.splitext(filename)[0]


            if filename not in anno_dict:
                print('pass, doesnt have anno info', filename)
                count += 1
                continue

            anno_info_list = anno_dict[filename]
            # definition annotation x1,y1,x2,y2 cls
            x1, y1, x2, y2, cls = anno_info_list

            #check labels(top_prediction) correct or wrong
            correct_label,wrong_label = classify_labels(top_prediction_label,cls,correct_label,wrong_label)
            print('correct_label',correct_label)
            print('wrong_label',wrong_label,)

            #extract top_prediction xyxy
            xyxy = top_prediction.convert('xyxy').bbox
            xyxyresult = extract_top_prediction(xyxy)

            #compute iou
            iou_result = bb_intersection_over_union(anno_dict[filename], xyxyresult)


            #save crop Image into crop_iamge as crop_img(folder)
            # crop_and_save_img(xyxyresult)

            #compute iousum
            iou_sum = iou_sum + iou_result
            iou_average = iou_sum / (wholedataset - count)
            #draw rectangle with prediction and anno rectagle
            for anno_info in anno_info_list:
                cv2.rectangle(predictions, (x1,y1), (x2,y2), color=(255,0,0))

            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            result_dir = os.path.join(result_path)

            #divide file by file_path like Type
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
            print('iou_sum:', iou_sum,'\n')
print('정답률:', correct_label/(correct_label + wrong_label))
print('iou_average:', iou_average)