import os

from PIL import Image
import cv2
from tools.img_utils import norm_path, split_path, image_list

IMAGE_SIZE = 512
image_dict = {}
mask_dict = {}


def make_dataset(data_dir, mask_dir, result_dir):
    # 1) read data dir
    for (path, dir, files) in os.walk(data_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext == '.png' or ext == '.jpg':
                mask_path = os.path.join(path, filename)

                # key: filename, value: image path
                image_dict[filename.replace(ext, '')] = mask_path

    # 2) read data dir
    for (path, dir, files) in os.walk(mask_dir):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext == '.jpg':
                mask_path = os.path.join(path, filename)

                # key: filename, value: image path
                mask_dict[filename.replace(ext, '')] = mask_path

    # 3) write nucleus style dataset
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    for id, image_path in image_dict.items():
        if not id in mask_dict:
            continue

        print('Processing {} ...'.format(id))
        mask_path = mask_dict[id]
        id_dir = os.path.join(result_dir, id)
        images_dir = os.path.join(id_dir, 'images')
        mask_dir = os.path.join(id_dir, 'masks')

        os.makedirs(images_dir)
        os.makedirs(mask_dir)

        # copy image
        im = Image.open(image_path)
        im.save(os.path.join(images_dir, id + '.png'))

        # copy mask
        im = Image.open(mask_path)
        im.save(os.path.join(mask_dir, id + '.png'))


def resize_image(im, height=IMAGE_SIZE, width=IMAGE_SIZE):
    def get_padding_size(im):
        h, w, _ = im.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(im)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # resize image
    resized_image = cv2.resize(constant, (height, width))
    return resized_image


def resize_image_file(img_path, image_size, save_path):
    img_path = (norm_path(img_path))
    save_path = norm_path(save_path)

    save_dir, _, _ = split_path(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read image
    im = cv2.imread(img_path)

    # resize image
    im = resize_image(im, image_size)

    # write image
    cv2.imwrite(save_path[:-3] + 'png', im)


def main(data_dir, result_dir, image_size):
    try:
        for file in image_list(data_dir):
            _, name, ext = split_path(file)
            save = os.path.join(result_dir, name + ext)
            print(save)
            resize_image_file(file, image_size, save)

        make_dataset(data_dir, mask_dir, result_dir)

    except IOError:
        pass  # You can always log it to logger


if __name__ == '__main__':
    data_dir = '~/data/Test_resized'
    mask_dir = '~/data/train1_resized_seg'
    result_dir = '~/data/mrcnn_cer/stage1_test'

    main(data_dir, result_dir, IMAGE_SIZE)

    data_dir = os.path.expanduser(data_dir)
    mask_dir = os.path.expanduser(mask_dir)
    result_dir = os.path.expanduser(result_dir)