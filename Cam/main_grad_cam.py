import argparse
import os
import torch
from collections import OrderedDict

from PIL import Image
from Utils.utils_gradcam import *

#from Model.densenet import DenseNet
from Official.densenet import DenseNet
from Model.resnet import ResNet
from Utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='~/data/mrcnn_cer/classificationdataset_224/test', help='Input image path')
parser.add_argument('--resume', default='/home/bong6/lib/robin_cer/results/classification_result_224_rotate/checkpoint_99.pth',
                    help='path to latest checkpoint')
parser.add_argument('--target_index', default=None, type=int, help='Target Index. ex) None, 0, 1, etc...')
parser.add_argument('--result', default='../result_gradcam', help='Output cam file dir name.')
parser.add_argument('--channels', default=3, type=int, help='select scale type rgb or gray')
parser.add_argument('--image_width', default=224, type=int, help='image crop width')
parser.add_argument('--image_height', default=224, type=int, help='image crop height')
parser.add_argument('--avg_pooling_width', default=7, type=int, help='average pooling width')
parser.add_argument('--avg_pooling_height', default=7, type=int, help='average pooling height')
parser.add_argument('--transparency', default=0.5, type=float, help='cam transparency')
parser.add_argument('--blur_times', default=1, type=int, help='cam blur_times')
parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
parser.add_argument('--densenet', default=True, action='store_true', help='set True to use densenet')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

# fix path
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


def main():
    if not os.path.isfile(args.resume):
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume) if use_cuda else torch.load(args.resume,
                                                                     map_location=lambda storage, loc: storage)

    avg_pool_size = (args.avg_pooling_height, args.avg_pooling_width)
    if args.densenet:
        # create Model
        model = DenseNet(num_init_features=32, growth_rate=16, block_config=(6, 12, 24, 16), channels=args.channels, avg_pooling_size=avg_pool_size, num_classes=args.num_classes)
        model = model.cuda() if use_cuda else model

        # create extractor
        extractor = DenseNetExtractor(model)
    else:
        # create Model
        model = ResNet(layers=[2, 2, 2, 2], channels=args.channels, global_pooling_size=avg_pool_size)
        model = model.cuda() if use_cuda else model

        # create extractor
        extractor = ResNetExtractor(model)

    # load Model
    state_dict = checkpoint['state_dict']
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        state_dict_rename[name] = v
    model.load_state_dict(state_dict_rename)
    print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # set Model for evaluation
    model.eval()

    # create gradient class activation map
    grad_cam = GradCam(extractor, use_cuda=use_cuda)

    for (path, dir, files) in os.walk(args.image_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
                image_path = os.path.join(path, filename)

                # read image
                img = pil_loader(image_path)
                img = np.float32(img) / 255.0
                input = preprocess_image(img, args.channels)

                # get class activation map
                cam, pred = grad_cam(input, args.target_index)

                cam = np.maximum(cam, 0)
                cam = cv2.resize(cam, (args.image_width, args.image_height))
                cam = cam - np.min(cam)
                mask = cam / np.max(cam)

                # make class activation map
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if args.channels == 1 else img
                result_img = make_cam_with_image(img, mask, transparency=args.transparency, blur_times=args.blur_times)

                # save cam image
                filename = os.path.splitext(filename)[0] + '@pred_' + str(pred)
                Image.fromarray(result_img).save(args.result + "/%s.png" % filename)
                print('Saved', image_path, '->', filename)


if __name__ == '__main__':
    main()
