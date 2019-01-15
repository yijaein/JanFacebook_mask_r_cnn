import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torchvision.models.densenet
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from Official.densenet import DenseNet
from Official.main import main
from Official.utils import AverageMeter
from Official.utils import accuracy
from Works.data_augmentation import *
from Works.utils import compute_auroc, softmax
from Official.utils import save_tensor_image
from Utils.auc_roc import save_auroc
from imgaug import augmenters as iaa

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/bong6/data/mrcnn_cer/classification_crop_hand_256', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--epoch_decay', default=100, type=int, help='learning rate decayed by 10 every N epochs')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay L2')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--pretrained', default=False, action='store_true', help='use pretrained model')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--evaluate', default=False, action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializingtraining')
parser.add_argument('--result', default='/home/bong6/lib/robin_cer/results/crop_random_9', help='path to result')
parser.add_argument('--aspect_ratio', default=False, action='store_true', help='keep image aspect ratio')
parser.add_argument('--resize_image_width', default=256, type=int, help='image width')
parser.add_argument('--resize_image_height', default=256, type=int, help='image height')
parser.add_argument('--image_width', default=224, type=int, help='image crop width')
parser.add_argument('--image_height', default=224, type=int, help='image crop height')
parser.add_argument('--avg_pooling_width', default=7, type=int, help='average pooling width')
parser.add_argument('--avg_pooling_height', default=7, type=int, help='average pooling height')
parser.add_argument('--channels', default=3, type=int, help='select scale type rgb or gray')
parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
parser.add_argument('--target_index', default=0, type=int, help='target index')
parser.add_argument('--classification_result', default='/home/bong6/lib/robin_cer/results/classification_Type', help='path for segmentation result')
parser.add_argument('--randomcrop', default=True, help='randomcrop')
parser.add_argument('--eval_result', default='/home/bong6/data/mrcnn_cer/classification_crop_hand_256/test', help='evaluate result folder path')
parser.add_argument('--save_tensor_image', default=False, help='check augmentation')
parser.add_argument('--save_auroc', default=False, help='save auroc graph')
args = parser.parse_args()

args.data = os.path.expanduser(args.data)
args.result = os.path.expanduser(args.result)
args.resume = os.path.expanduser(args.resume)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB') if args.channels == 3 else img.convert('L')
            out = pil_resize(img)
            return out

# set resize_image_size
def pil_resize(img):

    resize_image_size = (args.resize_image_width, args.resize_image_height)
    if args.aspect_ratio:
        img.thumbnail(resize_image_size)
        offset = ((resize_image_size[0] - img.size[0]) // 2, (resize_image_size[1] - img.size[1]) // 2)
        back = Image.new("RGB" if args.channels == 3 else 'L', resize_image_size, "black")
        back.paste(img, offset)
        out = back
    else:
        out = img.resize(resize_image_size)
    return out

 #add random_rotate and random_flip
def train_image_loader(path):

    out = pil_loader(path)
    out = np.array(out)
    # out = random_rotate(out, 360) #check
    # out = random_flip(out) #check

    #random Gausian_blur
    out = Random_Gausian_blur(out)

    out = Image.fromarray(np.uint8(out))

    return out


def valid_image_loader(path):
    out = pil_loader(path)
    return out


def log(*message, end='\n'):
    msg = ""
    for m in message:
        msg += str(m) + " "

    print(msg)
    file = os.path.join(args.result, 'detail_log.txt')
    with open(file, 'at') as f:
        f.write(msg + end)


def getImagesFiles(img_path):
    image_files = list()
    for (path, dir, files) in os.walk(img_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext != '.png' and ext != '.jpg':
                continue
            image_files.append(os.path.join(path, file))
    return image_files


class TrainDataset(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.Type_1 = 0
        self.Type_2 = 1
        self.Type_3 = 2

        # gathering crop_image and mask files
        # (sample_path, target)
        self.Type_1_samples = [[item, self.Type_1] for item in self.getType_1Files(root)]
        self.Type_2_samples = [[item, self.Type_2] for item in self.getType_2Files(root)]
        self.Type_3_samples = [[item, self.Type_3] for item in self.getType_3Files(root)]

        log('Train count samples', len(self.Type_1_samples), len(self.Type_2_samples), len(self.Type_3_samples))

        # balance ratio
        self.samples = self.Type_1_samples + self.Type_2_samples + self.Type_3_samples

    def balanceList(self, samples_list, ratio=[1.0, 1.0, 1.0]):
        major_samples = samples_list[0]
        minor_samples = samples_list[1:]

        # get count
        cnt_major_item = len(major_samples)

        # slice sample
        if ratio is not None:
            cnt_minor_item = [int(r * cnt_major_item) for r in ratio[1:]]
            for idx, samples in enumerate(minor_samples):
                n_slice = min(cnt_minor_item[idx], len(samples))
                random.shuffle(samples)
                minor_samples[idx] = samples[:n_slice]

        # concatenate sample
        balance_samples = list()
        balance_samples += major_samples
        msg = 'Train balance sample ' + str(len(balance_samples))
        for sam in minor_samples:
            random.shuffle(sam)
            balance_samples += sam
            msg += ' ' + str(len(sam))

        log(msg)

        return balance_samples

    def getType_1Files(self, path):

        get_Type_1_files = getImagesFiles(os.path.join(path, 'Type_1'))

        return get_Type_1_files

    def getType_2Files(self, path):

        get_Type_2_files = getImagesFiles(os.path.join(path, 'Type_2'))

        return get_Type_2_files

    def getType_3Files(self, path):

        get_Type_3_files = getImagesFiles(os.path.join(path, 'Type_3'))

        return get_Type_3_files

    def __getitem__(self, index):
        # shuffle no_seg_image rarely
        # if random.randint(0, len(self.Type_1_samples)) == 0:
        #     self.samples = self.balanceList([self.Type_1_samples, self.Type_2_samples, self.Type_3_samples])
        #
        #
        # self.samples = self.Type_1_samples + self.Type_2_samples + self.Type_3_samples
        sample_path, target = self.samples[index]

        sample = self.loader(sample_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, sample_path

    def __len__(self):
        return len(self.samples)


class ValDataset(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        self.root = root if not args.evaluate else args.data
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.Type_1 = 0
        self.Type_2 = 1
        self.Type_3 = 2

        if not args.evaluate:

            self.Type_1_samples = [[item, self.Type_1] for item in self.getType_1Files(self.root)]
            self.Type_2_samples = [[item, self.Type_2] for item in self.getType_2Files(self.root)]
            self.Type_3_samples = [[item, self.Type_3] for item in self.getType_3Files(self.root)]

            log('Val count samples', len(self.Type_1_samples), len(self.Type_2_samples), len(self.Type_3_samples))
            self.samples = self.Type_1_samples + self.Type_2_samples + self.Type_3_samples

        else:
            self.root = args.eval_result

            self.Type_1_samples = [[item, self.Type_1] for item in self.getType_1Files(self.root)]
            self.Type_2_samples = [[item, self.Type_2] for item in self.getType_2Files(self.root)]
            self.Type_3_samples = [[item, self.Type_3] for item in self.getType_3Files(self.root)]

            log('Test count samples', len(self.Type_1_samples), len(self.Type_2_samples), len(self.Type_3_samples))
            self.samples = self.Type_1_samples + self.Type_2_samples + self.Type_3_samples



    def getType_1Files(self, path):

        getType_1Files = getImagesFiles(os.path.join(path, 'Type_1'))

        return getType_1Files

    def getType_2Files(self, path):
        getTyp2_2Files = getImagesFiles(os.path.join(path, 'Type_2'))

        return getTyp2_2Files

    def getType_3Files(self, path):
        type3_files = getImagesFiles(os.path.join(path, 'Type_3'))

        return type3_files

    def __getitem__(self, index):
        sample_path, target = self.samples[index]

        sample = self.loader(sample_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, sample_path

    def __len__(self):
        return len(self.samples)


def pred(output, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        return pred


def train_model(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, input_path) in enumerate(train_loader):
        # measure data loading timepil_resize
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        #check agumentation
        if args.save_tensor_image:
            save_tensor_image(input, input_path, args.agumetation_check)

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
            message = ('Epoch: [{0}][{1}/{2}]\t' +
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' +
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' +
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})').format(epoch,
                                                                        i,
                                                                        len(train_loader),
                                                                        batch_time=batch_time,
                                                                        data_time=data_time,
                                                                        loss=losses,
                                                                        top1=top1)
            log(message)

def validate_model(val_loader, model, criterion, epoch, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cnt_cnt_label = [0] * args.num_classes
    cnt_exact_pred = [0] * args.num_classes

    dir_list = ['Type_1', 'Type_2', 'Type_3']
    # switch to evaluate mode
    model.eval()

    if args.evaluate:
        evaluate_csv_file = os.path.join(args.result, 'evaluate.csv')
        feval = open(evaluate_csv_file, 'wt')

    with torch.no_grad():
        end = time.time()
        target_index_output, target_index_target = list(), list()
        for i, (input, target, input_path) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # --------------------------------------
            # for auroc get value from target index
            output_cpu = output.squeeze().cpu().data.numpy()
            output_cpu = np.array([softmax(out)[args.target_index] for out in output_cpu])
            target_index_output.extend(output_cpu.astype(np.float))
            target_index_target.extend(np.equal(target.cpu().data.numpy(), args.target_index).astype(np.int))
            # --------------------------------------

            if args.evaluate:
                output_softmax = np.array([softmax(out) for out in output.cpu().numpy()])
                for file_path, pred_values in zip(input_path, output_softmax):
                    _, file = os.path.split(file_path)
                    name, _ = os.path.splitext(file)
                    line = ','.join([name] + [str(v) for v in pred_values])
                    feval.write(line + '\n')

            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].cpu().data.numpy()[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # put together for acc per label
            pred_list = pred(output).cpu().numpy().squeeze()
            target_list = target.cpu().numpy().squeeze()

            for (p, t) in zip(pred_list, target_list):
                cnt_cnt_label[t] += 1
                if p == t:
                    cnt_exact_pred[t] += 1

                pred_list = pred(output).cpu().numpy().squeeze()
                for pred_idx, pred_item in enumerate(pred_list):

                    dst = os.path.join(args.classification_result, dir_list[pred_item])

                    if not os.path.exists(dst):
                        os.makedirs(dst)

                    seg_img = input_path[pred_idx]
                    shutil.copy(seg_img, dst)

            if i % print_freq == 0:
                log(('Test: [{0}/{1}]\t' +
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' +
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t' +
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})').format(i,
                                                                      len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses,
                                                                      top1=top1))
                auc, roc = compute_auroc(target_index_output, target_index_target)
                if args.save_auroc:
                    save_auroc(auc, roc, os.path.join(args.result, str(epoch) + '.png'))


        log(' * Prec@1 {top1.avg:.3f} at Epoch {epoch:0}'.format(top1=top1, epoch=epoch))
        log(' * auc@1 {auc:.3f}'.format(auc=auc))

        for (i, (n_label, n_exact)) in enumerate(zip(cnt_cnt_label, cnt_exact_pred)):
            acc_label = (n_exact / n_label * 100) if n_label > 0 else 0
            log('acc of label {:0d}: {:0.3f}%'.format(i, acc_label))

    return auc


def densenet121_pretrained(**kwargs):
    # fix num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16)
    model = torchvision.models.densenet.densenet121(**kwargs)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, args.num_classes)
    )
    return model


if __name__ == '__main__':
    if args.channels == 3:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif args.channels == 1:
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    # create model
    avg_pool_size = (args.avg_pooling_height, args.avg_pooling_width)
    if args.pretrained:
        if args.channels != 3:
            print('(pretrained) Must fix channels == 3')
            exit()
        if args.avg_pooling_width != 7 and args.avg_pooling_height != 7:
            print('(pretrained) Must fix avg_pooling_size == 7')
            exit()
        # fix num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16)
        model = densenet121_pretrained(num_classes=args.num_classes)
    else:
        model = DenseNet(num_init_features=32, growth_rate=16, block_config=(6, 12, 24, 16),
                         num_classes=args.num_classes,
                         channels=args.channels, avg_pooling_size=avg_pool_size)


    if args.randomcrop:

        train_transforms = transforms.Compose([transforms.RandomCrop((args.image_height, args.image_width)),
                                               transforms.RandomResizedCrop(args.image_height, scale=(0.8, 1.0), ratio=(0.75, 1.3)),
                                               transforms.ToTensor(),
                                                normalize,
                                               ])

        val_transforms = transforms.Compose([transforms.CenterCrop((args.image_height, args.image_width)),
                                             transforms.ToTensor(),
                                                normalize,
                                             ])


    else:
        train_transforms = transforms.Compose([transforms.Resize(args.image_height),
                                               transforms.ToTensor(),
                                                normalize,
                                               ])

        val_transforms = transforms.Compose([transforms.Resize(args.image_height),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])


   # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # start main loop
    main(args, model, train_image_loader, valid_image_loader, normalize, optimizer,
         train_dataset=TrainDataset, valid_dataset=ValDataset,
         train_model=train_model, validate_model=validate_model,
         train_transforms=train_transforms,
         val_transforms=val_transforms,
         )
