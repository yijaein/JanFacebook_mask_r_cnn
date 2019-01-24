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



# def crop_and_save_img(xyxy):
#     x, y, w, h = xyxyresult
#
#     result_image = img[int(y):int(y + h), int(x):int(x + w)]
#     cv2.imwrite(crop_path, result_image)

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

def load_anno_dict(anno_file):
    anno_dict = dict()
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
            # print(filename, x, y, w, h, target)

            x1, y1 = int(x), int(y)
            w, h = int(w), int(h)

            x2 = x1 + w
            y2 = y1 + h

            if filename not in anno_dict:
                anno_dict[filename] = list()

            anno_dict[filename] = [(x1, y1, x2, y2), target]
    return anno_dict


def list_image(path, image_exts=['.png', '.jpg']):
    l = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in image_exts:
                continue
            else:
                l.append(os.path.join(root, file))
    return l



def detection(config_file, image_path, result_path, anno_file, crop_path):
    result = dict()

    crop_dir = os.path.join(result_path, 'crop')
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    anno_dict = load_anno_dict(anno_file)

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    coco_demo = RSNADemo(
        cfg,
        min_image_size=512,
        confidence_threshold=0.7,
    )

    for filePath in list_image(image_path):
        # read image
        img = cv2.imread(filePath, cv2.IMREAD_COLOR)

        # eval image
        result_image, x = coco_demo.run_on_opencv_image(img)

        # crop and save
        if crop_path:
            pass

        # draw GT bbox on result_image
        _, fileNameExt = os.path.split(filePath)
        fileName, ext = os.path.splitext(fileNameExt)
        if fileName in anno_dict:
            xyxy, cls = anno_dict[fileName]
            cv2.rectangle(result_image, xyxy[:2], xyxy[2:], color=(255, 0, 0))
        else:
            pass


        # save result image
        save_sub_path = [None, 'Type_1', 'Type_2', 'Type_3'][cls]
        save_path = os.path.join(result_path, save_sub_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, fileNameExt), result_image)

        result[fileName] = x

    return result



def eval_result(result, anno_file):
    anno_dict = load_anno_dict(anno_file)

    iou_list = []
    correct_label = 0
    wrong_label = 0
    for name, info in result.items():

        # if dont have anno
        if name not in anno_dict:
            continue

        top_pred = info['top_pred']
        xyxys = top_pred.convert('xyxy').bbox.numpy()
        if len(xyxys) == 0:
            continue
        xyxy = xyxys[0]

        labels = top_pred.get_field('labels').numpy()
        label = labels[0]

        xyxy_gt, label_gt = anno_dict[name]

        # compute iou
        iou = bb_intersection_over_union(xyxy_gt, xyxy)
        iou_list.append(iou)

        # compute acc label
        if label == label_gt:
            correct_label += 1
        else:
            wrong_label += 1

    print('(correct_label, wrong_label)',correct_label, wrong_label)
    print('acc:', correct_label/(correct_label+wrong_label))


    iou_average = sum(iou_list) / len(iou_list)
    print('iou_average',iou_average)


    # label acc

if __name__ == '__main__':
    config_file = "/home/bong6/lib/robin_cer/configs/e2e_mask_rcnn_R_50_FPN_1x_rsna.yaml"
    image_path = "/home/bong6/data/mrcnn_cer/stage1_train1/val2/images"

    anno_file = "/home/bong6/data/csv/output.csv"
    result_path = '/home/bong6/data/mrcnn_cer/result_rec'
    crop_path = ''

    do_eval_result = True

    result = detection(config_file, image_path, result_path, anno_file, crop_path)
    if do_eval_result:
        eval_result(result, anno_file)