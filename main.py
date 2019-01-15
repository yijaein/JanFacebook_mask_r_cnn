import argparse
import os
import time

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.tensor
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from Model.alexnet import AlexNet
from Model.densenet import DenseNet
from Model.resnet import ResNet
from Model.vggnet import vgg19
from Utils.auc_roc import save_auroc, compute_auroc, softmax
from Utils.utils import AverageMeter, adjust_learning_rate, save_checkpoint, accuracy, save_log, save_log_graph,save_loss_log

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/bong6/data/cervical_320', help='path to dataset')
parser.add_argument('--result', default='/home/bong6/lib/robin_intern/jiyi/result', help='path to result')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--evaluate', default=False, action='store_true', help='evaluate Model on validation set')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--image_width', default=320, type=int, help='image width')
parser.add_argument('--image_height', default=320, type=int, help='image height')
parser.add_argument('--resize_image_width', default=350, type=int, help='resize image width')
parser.add_argument('--resize_image_height', default=350, type=int, help='resize image height')
parser.add_argument('--global_pooling_width', default=10, type=int, help='global pooling width')
parser.add_argument('--global_pooling_height', default=10, type=int, help='global pooling height')
parser.add_argument('--densenet', default=True, action='store_true', help='set True to use densenet')
parser.add_argument('--num_classes', default=3, type=int, help='set classes number')
parser.add_argument('--target_index', default=0, type=int, help='target index')
parser.add_argument('--epoch_decay', default=70, type=int, help='learning rate decayed by 10 every N epochs')

args = parser.parse_args()

# expand path
args.data = os.path.expanduser(args.data)
args.result = os.path.expanduser(args.result)

best_prec1 = 0

if not os.path.exists(args.result):
    os.makedirs(args.result)


def main(model, criterion, optimizer, train_loader, val_loader):
    global best_prec1

    model = torch.nn.DataParallel(model).cuda()

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
            return

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.result)

        # save logZ
        log_file = save_log(epoch, prec1, args.result,  filename='log.txt')

        # save graph
        save_log_graph(log_file=log_file)
        adjust_learning_rate(args.learning_rate, optimizer, args.epochs, args.epoch_decay)



def train(train_loader, model, criterion, optimizer, epoch):
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

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record lossdks
        prec1 = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, top1=top1))

    savelogloss = save_loss_log(epoch, losses.avg, args.result,filename='losslog.txt')
    save_log_graph(log_file=savelogloss)



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        target_index_output, target_index_target = list(), list()
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # for auroc get value from target index
            output_cpu = output.cpu().data.numpy()

            output_cpu = np.array([softmax(out)[args.target_index] for out in output_cpu])

            target_index_output.extend(output_cpu.astype(np.float))
            target_index_target.extend(np.equal(target.cpu().data.numpy(), args.target_index).astype(np.int))
            # --------------------------------------

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                      loss=losses, top1=top1))

        auc, roc = compute_auroc(target_index_output, target_index_target)

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        save_auroc(auc, roc, os.path.join(args.result, 'rocgraph'+ '.png'))
    return top1.avg


if __name__ == '__main__':
    # 1) create Model
    global_pooling_size = (args.global_pooling_height, args.global_pooling_width)
    num_class = args.num_classes

    try:
        answer = int(input("어떤 모델로 학습시키시겠습니까? [1]DenseNet\t[2]ResNet\t[3]AlexNet\t[4]VGG\t[5]CANCEL\t"))

        if answer == 1:
            print('Create Densenet...')
            model = DenseNet(channels=3, block_config=(6, 12, 24, 16), global_pooling_size=global_pooling_size,
                             num_classes=num_class)  # Densenet 121
        elif answer == 2:
            print("create Resnet...")
            model = ResNet(layers=[2, 2, 2, 2], channels=3, global_pooling_size=global_pooling_size,
                           num_classes=num_class)  # resnet18
        elif answer == 3:
            print("create alexnet...")
            model = AlexNet(channels=3, num_classes=num_class)  # TODO: only 224 image can be handled
        elif answer == 4:
            print("create VGG...")
            model = vgg19(num_classes=num_class)  # TODO: only 224 image can be handled
        else:
            raise Exception('\n \t\t[1]DenseNet,\t[2]ResNet,\t[3]Alexnet\t,[4]VGG')

    except NameError as err:
        print("1 OR 2 만 입력해주세요 ")

    except ValueError as err:
        print("only number")

    except KeyboardInterrupt:
        print("retry")
    else:
        # 2) define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

        # 3) Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                #transforms.Resize((args.resize_image_height, args.resize_image_width)),
                transforms.RandomResizedCrop(args.image_width),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),  # convert int(0 ~ 255) -> float(0.0 ~ 1.0)
                normalize,
            ]))

        valid_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            #transforms.Resize((args.resize_image_height, args.resize_image_width)),
            #transforms.CenterCrop(args.image_width),
            transforms.ToTensor(),  # convert int(0 ~ 255) -> float(0.0 ~ 1.0)
            normalize,
        ]))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

        val_loader = torch.utils.data.DataLoader(valid_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)

        # 4) start main
        main(model, criterion, optimizer, train_loader, val_loader)
