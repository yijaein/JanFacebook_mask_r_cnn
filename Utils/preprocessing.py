import argparse
import os

from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', default='/home/bong6/data/train1/Type_1', help='Input image path')
parser.add_argument('--result', default='/home/bong6/data/cervical_320', help='Output cam file dir name.')
parser.add_argument('--image_width', default=320, type=int, help='image crop width')
parser.add_argument('--image_height', default=320, type=int, help='image crop height')
parser.add_argument('--channels', default=3, type=int, help='chaneels')

args = parser.parse_args()
args.image_path = os.path.expanduser(args.image_path)
args.result = os.path.expanduser(args.result)

if not os.path.isdir(args.result):
    os.makedirs(args.result)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB') if args.channels == 3 else img.convert('L')
            out = img.resize((args.image_width, args.image_height))

            return out


for (path, dir, files) in os.walk(args.image_path):
    for filename in files:
        # splitext directory ->file 경로를 나눈다
        ext = os.path.splitext(filename)[-1]
        if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
            image_path = os.path.join(path, filename)
            print(image_path)
            # read image
            img = pil_loader(image_path)
            print(img)

            img.save(args.result + "/%s.png" % filename)
            print('saved', image_path, '-->', filename, '&&', img)
