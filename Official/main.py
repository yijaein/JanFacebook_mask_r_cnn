import argparse
import random
import time
import warnings

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from Official.densenet import DenseNet
from Official.utils import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='~/data/256_ObjectCategories', help='path to dataset')
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
parser.add_argument('--result', default='../result', help='path to result')
parser.add_argument('--resize_image_width', default=256, type=int, help='image width')
parser.add_argument('--resize_image_height', default=256, type=int, help='image height')
parser.add_argument('--image_width', default=224, type=int, help='image crop width')
parser.add_argument('--image_height', default=224, type=int, help='image crop height')
parser.add_argument('--avg_pooling_width', default=7, type=int, help='average pooling width')
parser.add_argument('--avg_pooling_height', default=7, type=int, help='average pooling height')
parser.add_argument('--channels', default=3, type=int, help='select scale type rgb or gray')
parser.add_argument('--num_classes', default=256, type=int, help='number of classes')
args = parser.parse_args()

best_prec1 = 0


def train_model(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].cpu().data.numpy()[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch,
                                                                  i,
                                                                  len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time,
                                                                  loss=losses,
                                                                  top1=top1))


def validate_model(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].cpu().data.numpy()[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i,
                                                                      len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses,
                                                                      top1=top1))

        print(' * Prec@1 {top1.avg:.3f} at Epoch {epoch:0}'.format(top1=top1, epoch=epoch))

    return top1.avg


def main(args, model, train_image_loader, valid_image_loader, normalize, optimizer,
         train_dataset=datasets.ImageFolder, valid_dataset=datasets.ImageFolder,
         train_model=train_model, validate_model=validate_model, train_transforms=None, val_transforms=None):
    global best_prec1

    # save arguments
    save_args(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # parallel model
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    args.data = os.path.expanduser(args.data)
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # set transformation
    train_transforms = transforms.Compose([transforms.RandomCrop((args.image_height, args.image_width)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,
                                           ]) if train_transforms is None else train_transforms
    val_transforms = transforms.Compose([transforms.CenterCrop((args.image_height, args.image_width)),
                                         transforms.ToTensor(),
                                         normalize,
                                         ]) if val_transforms is None else val_transforms

    train_loader = torch.utils.data.DataLoader(train_dataset(traindir, transform=train_transforms, loader=train_image_loader),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset(valdir, transform=val_transforms, loader=valid_image_loader),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        validate_model(val_loader, model, criterion, 0, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args.lr, optimizer, epoch, args.epoch_decay)

        # train for one epoch
        train_model(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1 = validate_model(val_loader, model, criterion, epoch, args.print_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.result)
        save_accuracy(epoch + 1, prec1, args.result)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB') if args.channels == 3 else img.convert('L')
            img = img.resize((args.resize_image_width, args.resize_image_height))
            return img


if __name__ == '__main__':
    if args.channels == 3:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    # create model
    avg_pool_size = (args.avg_pooling_height, args.avg_pooling_width)
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=args.num_classes,
                     channels=args.channels, avg_pooling_size=avg_pool_size)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # start main loop
    main(args, model, pil_loader, pil_loader, normalize, optimizer)
