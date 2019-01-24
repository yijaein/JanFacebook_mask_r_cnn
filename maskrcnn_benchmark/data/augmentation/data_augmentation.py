import imgaug as ia
from imgaug import augmenters as iaa


def img_and_key_point_augmentation(augmentation, img, bbox, key_points):
    """
    Augment image and key points, bounding boxes !!
    :param augmentation: augmentation settings
    :param img: Only one image is needed. Not batch images
    :param bbox: [[x1,y1,x2,y2],[x1,y1,x2,y2]...]
    :param key_points: [[[x1,y1,x2,y2....],[x1,y1,x2,y2....]...]]
    :return: Returns augment image and key points, bbox
    """

    # img_copy = img.copy()
    image_shape = img.shape
    h, w = image_shape[0:2]

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    det = augmentation.to_deterministic()
    img_aug = det.augment_image(img)

    ia_bbox = list()
    for bounding_box in bbox:
        x1, y1, x2, y2 = bounding_box
        ia_bbox.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

    bbs = ia.BoundingBoxesOnImage(ia_bbox, shape=image_shape)
    bbs_aug = det.augment_bounding_boxes([bbs])[0]
    # img = bbs_aug.draw_on_image(img)

    after_bbox = list()
    for bounding_box in bbs_aug.bounding_boxes:
        bbox_list = [bounding_box.x1_int,bounding_box.y1_int,bounding_box.x2_int,bounding_box.y2_int]

        if bbox_list[0] >= w: bbox_list[0] = w-1
        if bbox_list[1] >= h: bbox_list[1] = h-1
        if bbox_list[2] >= w: bbox_list[2] = w-1
        if bbox_list[3] >= h: bbox_list[3] = h-1

        if bbox_list[0] == bbox_list[2] or bbox_list[1] == bbox_list[3]:
            return img_and_key_point_augmentation(augmentation, img, bbox, key_points)

        bbox_list = list(map(lambda x: max(x, 0), bbox_list))
        after_bbox.append(bbox_list)

    after_key_points = list()
    for key_point_list in key_points:
        after_key_point_list = list()
        for key_point in key_point_list:
            xy_points = list()
            for i, x in enumerate(key_point[::2]):
                y = key_point[(i*2)+1]
                xy_points.append(ia.Keypoint(x=x, y=y))

            keypoints_on_image = det.augment_keypoints([ia.KeypointsOnImage(xy_points, shape=image_shape)])
            # img = keypoints_on_image[0].draw_on_image(img)

            xy_points = list()
            for key_point in keypoints_on_image[0].keypoints:
                kp = [key_point.x_int, key_point.y_int]
                if 0 > min(kp) or w <= max(kp[::2]) or h <= max(kp[1::2]):
                    # print(kp)
                    return img_and_key_point_augmentation(augmentation, img, bbox, key_points)
                xy_points.extend(kp)

            after_key_point_list.append(xy_points)

        after_key_points.append(after_key_point_list)

    assert img_aug.shape == image_shape, "Augmentation shouldn't change image size"

    return img_aug, after_bbox, after_key_points


def img_augmentation(augmentation, img, bbox):
    """
    Augment image and bounding boxes !!
    :param augmentation: augmentation settings
    :param img: Only one image is needed. Not batch images
    :param bbox: [[x1,y1,x2,y2],[x1,y1,x2,y2]...]
    :return: Returns augment image and bbox
    """

    # img_copy = img.copy()
    image_shape = img.shape
    h, w = image_shape[0:2]

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    det = augmentation.to_deterministic()
    img_aug = det.augment_image(img)

    ia_bbox = list()
    for bounding_box in bbox:
        x1, y1, x2, y2 = bounding_box
        ia_bbox.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))

    bbs = ia.BoundingBoxesOnImage(ia_bbox, shape=image_shape)
    bbs_aug = det.augment_bounding_boxes([bbs])[0]
    # img = bbs_aug.draw_on_image(img)

    after_bbox = list()
    for bounding_box in bbs_aug.bounding_boxes:
        bbox_list = [bounding_box.x1_int,bounding_box.y1_int,bounding_box.x2_int,bounding_box.y2_int]

        if bbox_list[0] >= w: bbox_list[0] = w-1
        if bbox_list[1] >= h: bbox_list[1] = h-1
        if bbox_list[2] >= w: bbox_list[2] = w-1
        if bbox_list[3] >= h: bbox_list[3] = h-1

        if bbox_list[0] == bbox_list[2] or bbox_list[1] == bbox_list[3]:
            return img_augmentation(augmentation, img, bbox)

        bbox_list = list(map(lambda x: max(x, 0), bbox_list))
        after_bbox.append(bbox_list)

    assert img_aug.shape == image_shape, "Augmentation shouldn't change image size"

    return img_aug, after_bbox


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    import random

    img = cv2.imread("/home/bong04/data/kmu/resize_image_896_2048/train/org_896_2048/2167971.png")

    x1 = 61
    y1 = 132
    w = 123
    h = 263
    x2 = x1 + w
    y2 = y1 + h

    key_points = [[[x1, y1, x1, y2, x2, y2, x2, y1], [x1, y1, x1, y2, x2, y2, x2, y1]]]
    bbox = [[x1, y1, x2, y2]]

    x1 = 299
    y1 = 137
    w = 110
    h = 157
    x2 = x1 + w
    y2 = y1 + h

    key_points.append([[x1, y1, x1, y2, x2, y2, x2, y1], [x1, y1, x1, y2, x2, y2, x2, y1]])
    bbox.append([x1, y1, x2, y2])

    print(random.randint(0, 10000))
    ia.seed(random.randint(0, 10000))

    augmentation = iaa.SomeOf((1, None), [
        # iaa.Fliplr(0.5),
        iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-40, 40),

            # shear=(-8, 8)
        ),
        iaa.AverageBlur(
            k=(2, 11)
        ),
        iaa.Crop(px=(224,224))
        # iaa.Multiply((0.9, 1.1))
    ])

    print(bbox, key_points)
    img, bbox, key_points = img_and_key_point_augmentation(augmentation, img, bbox, key_points)
    print(bbox, key_points)

    def imshow(img):
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.axis("off")
        plt.waitforbuttonpress()

    imshow(img)