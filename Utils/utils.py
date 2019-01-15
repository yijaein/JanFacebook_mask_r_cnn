import os
import shutil

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from matplotlib import pyplot as plt
from torch.autograd import Variable


def save_checkpoint(state, is_best, result_path, checkpoint='checkpoint.pth', model_best='model_best.pth'):
    # check result1 directory exist
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    # save Model
    torch.save(state, os.path.join(result_path, checkpoint))
    if is_best:
        shutil.copyfile(os.path.join(result_path, checkpoint), os.path.join(result_path, model_best))


# after dinner add -> loss value
def save_log(epoch, prec, result_path, filename='log1.txt', sep='\t'):
    # check result1 directory exist
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    log_file = os.path.join(result_path, filename)
    with open(log_file, 'at', encoding='utf8') as f:
        f.write(str(epoch) + sep + truncate(prec, 4) + '\n')
    return log_file

def save_loss_log(epoch,loss, result_path, filename='losslog.txt', sep ='\t'):
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    log_file = os.path.join(result_path,filename)
    with open(log_file, 'at', encoding='utf8') as f:
        f.write(str(epoch)+sep+truncate(loss, 4)+ '\n')
    return log_file


def save_log_graph(log_file, sep='\t', x_column_index=0, y_column_index=1):
    if not os.path.isfile(log_file):
        print('log file not found:', log_file)
        return

    path, file = os.path.split(log_file)
    name, _ = os.path.splitext(file)
    save_file = os.path.join(path, name + '.png')

    xValue, yValue = [], []
    with open(log_file, 'rt', encoding='utf8') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            columns = line.replace('\n', '').split(sep)
            if len(columns) < 2:
                continue

            xValue.append(float(columns[x_column_index]))  # accuracy
            yValue.append(float(columns[y_column_index]))  # epoch

    # 3) save graph
    plt.plot(xValue, yValue)
    plt.title(name)
    plt.savefig(save_file)
    plt.close()


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


def adjust_learning_rate(learning_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res_ = correct_k.mul_(100.0 / batch_size).cpu().numpy()[0]
            res.append(res_)
        return res


# utils
def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


def preprocess_image(img, channels):
    if channels == 3:
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        preprocessed_img = img.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    elif channels == 1:
        preprocessed_img = img.copy()

        preprocessed_img[:, :, ] = preprocessed_img[:, :, ] - 0.5
        preprocessed_img[:, :, ] = preprocessed_img[:, :, ] / 0.5

        preprocessed_img = np.expand_dims(preprocessed_img, axis=2)

    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)

    # convert to tensor
    input = Variable(preprocessed_img, requires_grad=True)
    return input

def adjust_learning_rate(lr, optimizer, epoch, decay_epoch=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#not yet
def save_log_graph_type3_val(path, xValue, yValue):
    save_file = os.path.join(path)
    xValue, yValue = [],[]


    # 3) save graph
    #type_1
    plt.plot(xValue, yValue, 'r', label='Type_1')
    plt.title('label acc')

    #type_2
    plt.plot(xValue, yValue, 'g', label='Type_2')


    #type_3
    plt.plot(xValue, yValue, 'b', label='Type_3')
    plt.legend()
    plt.savefig(save_file)
    plt.close()
