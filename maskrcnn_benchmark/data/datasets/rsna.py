import os
import csv
import cv2
import numpy as np

import torch
import torch.utils.data
from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt
from imgaug import parameters as iap
import random

class RSNADataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "Type_1",
        "Type_2",
        "Type_3"
    )

    def __init__(self, ann_file, root, remove_images_without_annotations, mask_type='polygon', transforms=None):
        # "mask_type" = "polygon" or "image"

        self.mask_type = mask_type
        self.img_key_list = list()
        self.img_dict = dict()
        self.ann_info = list()

        cls = RSNADataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

        img_dict = dict()
        for dirName, subdirList, fileList in os.walk(root):
            for filename in fileList:
                filename, ext = os.path.splitext(filename)
                if ext.lower() in [".png", ".jpg", ".jpeg"]:
                    img_dict[filename] = os.path.join(root, dirName, filename + ext)

        with open(ann_file, 'r') as ann_f:
            ann_cvf = csv.reader(ann_f)

            for i, line in enumerate(ann_cvf):
                if i == 0:
                    continue

                filename, x, y, w, h, cls = line

                if remove_images_without_annotations:
                    if cls == 0:
                        continue

                    x1 = int(x)
                    y1 = int(y)
                    x2 = int(w) + x1
                    y2 = int(h) + y1

                else:
                    x1 = 0
                    y1 = 0
                    x2 = 0
                    y2 = 0

                try:
                    self.ann_info.append(([x1, y1, x2, y2, cls], img_dict[filename]))

                except KeyError:
                    continue

        # 중복 방지 RSNA csv 파일 참조
        # self.img_key_list = list(set(self.img_key_list))
        self.transforms = transforms

    def __getitem__(self, idx):
        ann_info, image_path = self.ann_info[idx]
        # read image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # gen target
        height, width = img.shape[:2]
        target = self.get_groundtruth(ann_info, width, height)
        target = target.clip_to_image(remove_empty=True)

        # augment image and target
        img, target = self.apply_iaa(img, target)


        img = Image.fromarray(img, mode="RGB")
        # plt.imshow(img)
        # plt.show()
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx


    def __len__(self):
        return len(self.ann_info)


    def apply_iaa(self, image, target):

        numbbox = target.bbox.numpy()
        bboxresult = []

        for item in numbbox:
            bboxresult.extend(item)

        a, b, c, d = bboxresult
        # =================check bounding box======================================================

        # image_debug = image.copy()
        # image_debug = cv2.rectangle(image_debug, (int(a), int(b)), (int(c), int(d)), color=(255, 0, 0))
        # plt.imshow(image_debug)
        # plt.show()


        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=a, y1=b, x2=c, y2=d)
        ], shape=image.shape[:2])

        seq = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.Fliplr(0.5),
                          iaa.GaussianBlur(sigma=1.5)
                          ),

            # iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
            iaa.Affine(
                # translate_px={"x": 40, "y": 60},
                # rotate=(-40, 40),
                scale=(0.8, 1.3)
            ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs

            iaa.SomeOf(1,[
                           iaa.Crop(px=(0, 50), random_state=True),
                           # CenterCrop(450+random.randrange(0,20), 450+random.randrange(0,20))
                       ])


        ],random_order=True)




        seq_det = seq.to_deterministic()

        image_aug = seq_det.augment_image(image)
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        x1, y1, x2, y2 = bbs_aug.bounding_boxes[0].x1, bbs_aug.bounding_boxes[0].y1, bbs_aug.bounding_boxes[0].x2, bbs_aug.bounding_boxes[0].y2
        target.bbox = torch.as_tensor([[x1, y1, x2, y2]], dtype=torch.float32, device=target.bbox.device)
        return image_aug, target


    def get_groundtruth(self, ann_info, width, height):
        # anno = self._preprocess_annotation(self.ann_info[filename], width, height)

        x1, y1, x2, y2, cls = ann_info

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cls = int(cls)

        box = [x1, y1, x2, y2]
        target = BoxList(torch.tensor([box]), (width, height), mode="xyxy")
        target.add_field("labels", torch.tensor([cls]))

        # masks = SegmentationMask(anno["masks"], (width, height))
        # masks = SegmentationMask(anno["masks"], (width, height), type=self.mask_type)
        # target.add_field("masks", masks)

        return target

    def _preprocess_annotation(self, target, width, height):
        boxes = []
        temp_masks= []
        masks = []
        gt_classes = []

        for ann_info in target:
            if ann_info == None:
               continue
            mask = np.zeros((height, width))

            bndbox = ann_info[:4]
            mask[bndbox[0]:bndbox[2],bndbox[1]:bndbox[3]] = 1

            x1, y1, x2, y2 = ann_info[:4]
            temp_mask = [[x1,y1,x1,y2,x2,y2,x2,y1]]
            temp_masks.append(temp_mask)

            boxes.append(bndbox)
            masks.append([mask])

            # 만약 클래스가 번호가 아닌 이름으로 있다면 아래 코드를 사용한다.
            # gt_classes.append(self.class_to_ind[ann_info[-1]])
            gt_classes.append(ann_info[-1])

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            # "masks": masks,
            "masks": temp_masks,
            "labels": torch.tensor(gt_classes),
        }
        return res

    def _get_image_polygons(self, mask):
        _, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return 0

    def get_img_info(self, index):
        return {"height": 512, "width": 512}

    def map_class_id_to_class_name(self, class_id):
        return RSNADataset.CLASSES[class_id]

def CenterCrop(height, width):
    crop = iaa.CropToFixedSize(height=height, width=width)
    crop.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    pad = iaa.PadToFixedSize(height=height, width=width, pad_mode=ia.ALL, pad_cval=(0, 255))
    pad.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    return iaa.Sequential([crop, pad])
