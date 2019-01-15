import argparse
from collections import OrderedDict

from Model.densenet import DenseNet
from Model.resnet import ResNet
from Utils.utils import *
from Utils.utils_gradcam import *
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', default='~/data/cervical_320', help='Input image path')
parser.add_argument('--result', default='../result_cam', help='Output cam file dir name.')
parser.add_argument('--image_width', default=320, type=int, help='image crop width')
parser.add_argument('--image_height', default=320, type=int, help='image crop height')
parser.add_argument('--channels', default=3, type=int, help='chaneels ')
parser.add_argument('--resume', default='/home/bong6/lib/robin_intern/jiyi/result/model_best.pth')
parser.add_argument('--densenet', default=True, action='store_true', help='set True to use densenet')
parser.add_argument('--avg_pooling_width', default=10, type=int, help='average pooling width')
parser.add_argument('--avg_pooling_height', default=10, type=int, help='average pooling height')
parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
parser.add_argument('--transparency', default=0.9, type=float, help='cam transparency')
parser.add_argument('--blur_times', default=1, type=int, help='cam blur_times')
args = parser.parse_args()

use_cuda = False  # torch.cuda.is_available()

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


def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample=(224, 224)):
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)

    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def main():
    if not os.path.isfile(args.resume):
        raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume) if use_cuda else torch.load(args.resume,
                                                                     map_location=lambda storage, loc: storage)

    avg_pool_size = (args.avg_pooling_height, args.avg_pooling_width)
    if args.densenet:
        # create Model
        model = DenseNet(channels=args.channels, global_pooling_size=avg_pool_size, num_classes=args.num_classes)
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

    # get weight only from the last layer(linear)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    # set Model for evaluation
    model.eval()

    for (path, dir, files) in os.walk(args.image_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png' or ext == '.jpg' or ext == '.jpeg':
                image_path = os.path.join(path, filename)

                # read image
                img = pil_loader(image_path)
                img = np.float32(img) / 255.0
                input = preprocess_image(img, args.channels)

                # extract features and output
                features, output = extractor(input)
                features, output = features[0].cpu().data.numpy(), output.cpu().data.numpy()

                # get prediction index
                pred = np.argmax(output)

                image_size = (args.image_width, args.image_height)
                CAMs = returnCAM(features, weight_softmax, [pred], size_upsample=image_size)
                # 0~255

                CAM = np.array(CAMs)[0]

                CAM = CAM - np.min(CAM)
                mask = CAM / np.max(CAM)
                img, mask = np.uint8(255 * img), np.uint8(255 * mask)

                heatmap = heat_map_overlay(img, mask, transparency=args.transparency, blur_times=args.blur_times)

                filename = os.path.splitext(filename)[0] + '@pred_' + str(pred)
                Image.fromarray(heatmap).save(args.result + "/%s.png" % filename)
                print('Saved', image_path, '->', filename)


if __name__ == '__main__':
    main()
