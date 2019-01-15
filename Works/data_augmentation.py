import random

import cv2
import numpy as np
from PIL import ImageFilter


####################
# random blur
####################
def random_blur(image, blur_radius):
    return image.filter(ImageFilter.GaussianBlur(radius=random.random() * blur_radius))


####################
# random clahe
####################
def apply_clahe(image, clip_limit):
    if len(image.shape) == 2:
        shape = image.shape
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        image = clahe.apply(image)
    elif len(image.shape) == 3:
        shape = image.shape
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        image = bgr
    return image.reshape(shape)


def random_clahe(image, min, max):
    do_apply = random.randint(0, 1)
    if do_apply == 1:
        clip_limit = min + (max - min) * random.random()
        image = apply_clahe(image.astype("uint8"), clip_limit=clip_limit)
    return image


####################
# random gamma
####################
def adjust_gamma(image, gamma=1.0):
    shape = image.shape
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table).reshape(shape)


def random_gamma(image, min, max):
    gamma = random.uniform(min, max)
    return adjust_gamma(image.astype("uint8"), gamma=gamma)


####################
# random sharpen
####################
def sharpen(image, ratio=0.5):
    shape = image.shape
    # image = image.reshape((shape[0], shape[1]))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    unsharp_image = cv2.filter2D(image, -1, kernel)
    unsharp_image = image * (1.0 - ratio) + unsharp_image * ratio
    unsharp_image = unsharp_image.reshape(shape)
    return unsharp_image


def random_sharpen(image, max_ratio=0.5):
    do_sharp = random.randint(0, 1)
    if do_sharp > 0:
        ratio = random.uniform(0.001, max_ratio)
        image = sharpen(image, ratio=ratio)
        image[image > 255] = 255.0
        image[image < 0] = 0.0
    return image



def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def random_rotate(image, max_ratio):
    ratio = random.uniform(-max_ratio, max_ratio)
    image = rotate(image, ratio, center=None, scale = 1.0)

    return image

def random_flip(image):
    ratio = random.randint(0,1)
    img = cv2.flip(image, ratio)

    return img

#not used yet
def randomCrop(img, width, height):
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

#ratio must within 5
def Random_Gausian_blur(img):

    x = random.randint(0,10)
    if x % 2 == 0:
        x = 0

    else:
        #bgr convert rgb
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        # GaussianBlur
        img = cv2.GaussianBlur(img, (10 - x, 10 - x), 0)
    return img