import math
import os

import cv2
import numpy as np

save_crop_seg_path = '~/data/CropKidneySeg'
save_crop_seg_path = os.path.expanduser(save_crop_seg_path)
save_crop_seg_image = True

if save_crop_seg_image and not os.path.isdir(save_crop_seg_path):
    os.makedirs(save_crop_seg_path)

padding_size = 20
seg_padding_size = 0
equal_histogram = False
do_preprocess = False
do_seg = False

# create CLAHE
clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))


def main(image_path, seg_path, output_path):
    output_path = output_path.replace('CropKidney', 'CropKidneyHist') if equal_histogram else output_path
    output_path = output_path.replace('CropKidney', 'CropKidneyColor') if do_preprocess else output_path
    output_path = output_path.replace('CropKidney', 'CropKidneySeg') if do_seg else output_path

    # expand user
    image_path = os.path.expanduser(image_path)
    seg_path = os.path.expanduser(seg_path)
    output_path = os.path.expanduser(output_path)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 1) Read images
    data_input = dict()
    for (path, dir, files) in os.walk(image_path):
        for filename in files:
            if filename.endswith('.png'):
                data_input[filename] = os.path.join(path, filename)

    cnt_black_seg = 0
    for (path, dir, files) in os.walk(seg_path):
        for filename in files:
            if filename.endswith('.png'):
                if not filename in data_input:
                    continue

                # read image
                input_image = cv2.imread(data_input[filename])
                seg_image = cv2.imread(os.path.join(path, filename))
                seg_image = cv2.resize(seg_image, (input_image.shape[1], input_image.shape[0]))

                if seg_image.sum() == 0:
                    cnt_black_seg += 1
                    print(filename, 'is black segmentation, Pass!')
                    continue

                # convert RGB to gray
                seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

                # add pad to seg image
                if seg_padding_size > 0:
                    _, contours, hierachy = cv2.findContours(seg_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    seg_image = cv2.drawContours(seg_image, contours, -1, 255, seg_padding_size)

                # fill seg image inside
                seg_image = contour_filling(seg_image)

                # do equal histogram
                if equal_histogram:
                    input_image = clahe.apply(input_image)

                # do segmentation
                if do_seg:
                    input_image = input_image * (seg_image / 255.0)

                # rotate image
                angle = calculate_angle(seg_image)
                seg_image = rotate_bound(seg_image, angle)
                input_image = rotate_bound(input_image, angle)

                # Add black padding to input image and seg image
                seg_image = cv2.copyMakeBorder(seg_image, padding_size, padding_size, padding_size, padding_size, 0)
                input_image = cv2.copyMakeBorder(input_image, padding_size, padding_size, padding_size, padding_size, 0)

                # get white pixel bounding box
                x, y, w, h = find_bounding_square(seg_image, padding=padding_size)


                result_image = input_image[int(y):int(y + h), int(x):int(x + w)]

                if do_preprocess:
                    result_seg_image = seg_image[int(y):int(y + h), int(x):int(x + w)]
                    result_image = preproess_image(result_image, result_seg_image)

                if save_crop_seg_image:
                    seg_image_crop = seg_image[int(y):int(y + h), int(x):int(x + w)]
                    cv2.imwrite(os.path.join(save_crop_seg_path, filename), seg_image_crop)

                cv2.imwrite(os.path.join(output_path, filename), result_image)
                print(filename, ' --- ', input_image.shape, x, y, w, h)

    print('num of uncroped black seg images:', cnt_black_seg)


def calculate_angle(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    rows, cols = img.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)

    angle = math.degrees(math.atan2(lefty - righty, rows))

    return angle


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def contour_filling(image):
    image = image.astype(np.uint8)

    _, contours, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    areaArray = []

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    # first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    for i in reversed(range(len(areaArray))):
        contour = sorteddata[i][1]
        if i < 2:
            cv2.drawContours(image, [contour], 0, 255, -1)
        else:
            cv2.drawContours(image, [contour], 0, 0, -1)

    return np.array(image)


def find_bounding_square(image, padding=0):
    image = image.astype(np.uint8)

    # get shape
    height, width = image.shape
    x1 = width
    y1 = height
    x2 = 0.0
    y2 = 0.0

    _, contours, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if x < x1: x1 = x
        if y < y1: y1 = y
        if (x + w) > x2: x2 = x + w
        if (y + h) > y2: y2 = y + h

    w, h = x2 - x1, y2 - y1

    x, y, w_padded, h_padded = (x1 - padding, y1 - padding, w + padding * 2, h + padding * 2)

    # correct figures (Problems arising from padding, for example -1)
    x = x if x > 0 else 0
    y = y if y > 0 else 0

    return x, y, w_padded, h_padded


def image_ori(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img


def image_dt(img, seg_image):
    img = img * (seg_image / 255.0)
    img = img.astype(np.uint8)

    img_canny = cv2.Canny(img, 100, 200)
    img_canny = cv2.bitwise_not(img_canny)
    img_dt = cv2.distanceTransform(img_canny, cv2.DIST_L2, 0);

    img_dt = img_dt * (seg_image / 255.0)
    img_dt = img_dt.astype(np.uint8)

    cv2.normalize(img_dt, img_dt, 0, 255, cv2.NORM_MINMAX)
    return img_dt


def image_grad(img, seg_image):
    img_gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    img_gradx = np.absolute(img_gradx)

    img_grady = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    img_grady = np.absolute(img_grady)
    img_grad = img_gradx + img_grady

    img_grad = img_grad * (seg_image / 255.0)

    cv2.normalize(img_grad, img_grad, 0, 255, cv2.NORM_MINMAX)
    return img_grad


def image_merge(img_r, img_g, img_b):
    img_pseudo = np.dstack((img_r, img_g, img_b))
    img_pseudo = img_pseudo.astype(np.uint8)
    return img_pseudo


def preproess_image(img, seg_image):
    img_ori = image_ori(img)
    img_dt = image_dt(img, seg_image)
    img_grad = image_grad(img, seg_image)

    img_pseudo = image_merge(img_ori, img_grad, img_dt)
    return img_pseudo


if __name__ == '__main__':
    main(image_path='~/data/180718_KidneyUS_400_png/aki',
         seg_path='~/data/180718_KidneyUS_400_png_seg',
         output_path='~/data/CropKidney/train/normal')

    main(image_path='~/data/180718_KidneyUS_400_png/ckd',
         seg_path='~/data/180718_KidneyUS_400_png_seg',
         output_path='~/data/CropKidney/train/ckd')

    main(image_path='~/data/180718_KidneyUS_400_png/normal',
         seg_path='~/data/180718_KidneyUS_400_png_seg',
         output_path='~/data/CropKidney/train/normal')

    main(image_path='~/data/180725_KidneyUS_100_png/aki',
         seg_path='~/data/180725_KidneyUS_100_png_seg',
         output_path='~/data/CropKidney/val/normal')

    main(image_path='~/data/180725_KidneyUS_100_png/ckd',
         seg_path='~/data/180725_KidneyUS_100_png_seg',
         output_path='~/data/CropKidney/val/ckd')

    main(image_path='~/data/180725_KidneyUS_100_png/normal',
         seg_path='~/data/180725_KidneyUS_100_png_seg',
         output_path='~/data/CropKidney/val/normal')
