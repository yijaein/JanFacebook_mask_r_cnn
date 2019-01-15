import argparse

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from Official.main import main

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_name_help = 'model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='~/data/256_ObjectCategories', help='path to dataset')
parser.add_argument('--arch', default='resnet18', choices=model_names, help=model_name_help)
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--epoch_decay', default=40, type=int, help='learning rate decayed by 10 every N epochs')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--evaluate', default=False, action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--result', default='../result_pretrained', help='path to result')
parser.add_argument('--resize_image_width', default=256, type=int, help='image width')
parser.add_argument('--resize_image_height', default=256, type=int, help='image height')
args = parser.parse_args()

# Should fixed settings
args.image_width = 224
args.image_height = 224
args.num_classes = 1000
args.channels = 3


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB') if args.channels == 3 else img.convert('L')
            img = img.resize((args.resize_image_width, args.resize_image_height))
            return img


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # create model
    model = models.__dict__[args.arch](num_classes=args.num_classes, pretrained=True)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # start main loop
    main(args, model, pil_loader, pil_loader, normalize, optimizer)
