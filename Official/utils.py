import os
import shutil
import numpy as np
import torch
from tools.img_utils import norm_path, split_path, image_list
import scipy.misc
count = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, optimizer, epoch, decay_epoch=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_args(args):
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    with open(os.path.join(args.result, 'args.txt'), 'w') as f:
        line = str(args).replace('Namespace(', '').replace(')', '')
        for l in line.split(', '):
            f.write(l + '\n')


def save_checkpoint(state, is_best, result_dir):
    filename = os.path.join(result_dir, 'checkpoint_{}.pth'.format(state['epoch']))
    checkpoint_dir = '/'.join(filename.split('/')[:-1])
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(result_dir, 'model_best.pth'))


def save_accuracy(epoch, accuracy, result_dir):
    filename = os.path.join(result_dir, 'log.txt')
    result_dir = '/'.join(filename.split('/')[:-1])
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    with open(filename, 'at') as f:
        f.write('{}\t{:.3f}\n'.format(str(epoch), accuracy))


def get_increasing_filename():
    global count
    count += 1
    return str(count) + '.png'

#check agumentation result
def save_tensor_image(input, filename_list=None, path='temp'):

    if not os.path.isdir(path):
        os.makedirs(path)

    image_list = input.cpu().data.numpy()

    for i, image in enumerate(image_list):
        # get filename
        filename = get_increasing_filename() if filename_list is None else filename_list[i]

        #print(image.shape)

        image = np.transpose(image, axes=[1, 2, 0])
        #print(image.shape)
        _,file_dir,_ = split_path(filename)

        # save image

        scipy.misc.imsave(os.path.join(path, file_dir + '.png'), image.squeeze())