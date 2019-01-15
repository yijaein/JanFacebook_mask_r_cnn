import os

from PIL import Image

if __name__ == '__main__':
    data_dir = '~/data/Test_resized'
    mask_dir = '~/data/train1_resized_seg'
    result_dir = '~/data/mrcnn_cer/stage1_test'

    data_dir = os.path.expanduser(data_dir)
    mask_dir = os.path.expanduser(mask_dir)
    result_dir = os.path.expanduser(result_dir)

    image_dict = {}
    mask_dict = {}

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