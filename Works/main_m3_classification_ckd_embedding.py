import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from Official.main import main
from Official.utils import AverageMeter
from Official.utils import accuracy
from Works.utils import compute_auroc, softmax

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/bong07/data/yonsei2/machine/dataset/kidney_patient_info(동일항목수정)', help='path to dataset')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
parser.add_argument('--lr', default=1e-1, type=float, help='initial learning rate')
parser.add_argument('--epoch_decay', default=500, type=int, help='learning rate decayed by 10 every N epochs')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
parser.add_argument('--pretrained', default=False, action='store_true', help='use pretrained model')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--evaluate', default=False, action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--result', default='/home/bong07/data/yonsei2/machine/model/m3(동일항목수정)_seed30', help='path to result')
parser.add_argument('--aspect_ratio', default=False, action='store_true', help='keep image aspect ratio')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
parser.add_argument('--input_shape', default=6, type=int, help='number of input shape')
parser.add_argument('--target_index', default=1, type=int, help='target index')
parser.add_argument('--classification_result', default='', help='path for segmentation result')
args = parser.parse_args()

args.data = os.path.expanduser(args.data)
args.result = os.path.expanduser(args.result)
args.result = os.path.abspath(args.result)
args.resume = os.path.expanduser(args.resume)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, padding_idx, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param padding_idx: index of the PAD token
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        non_pad_mask = (target != self.padding_idx)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)[non_pad_mask]
        smooth_loss = -logprobs.mean(dim=-1)[non_pad_mask]
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.sum()


def gradClamp(parameters, clip=5):
    for p in parameters:
        p.grad.data.clamp_(min=-clip, max=clip)


criterion_smoothing = LabelSmoothing(padding_idx=-100, smoothing=0.1)
grad_clip = 1.0


def dummyImageLoader(path):
    return None


class DummyDataset(Dataset):
    def __init__(self, root, loader, transform=None, target_transform=None):
        pass

    def __getitem__(self, index):
        return None

    def __len__(self):
        return 0


def find_csv(path, sub='train'):
    if sub:
        path = os.path.join(path, sub)

    for (root, dirs, files) in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in ['.csv']:
                return os.path.join(root, file)


def log(*message, end='\n'):
    msg = ""
    for m in message:
        msg += str(m) + " "

    print(msg)
    file = os.path.join(args.result, 'detail_log.txt')
    with open(file, 'at') as f:
        f.write(msg + end)


def save_values(epoch, loss, accuracy, auc, acc_label_list):
    filename = os.path.join(args.result, 'log_values.txt')

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    with open(filename, 'at') as f:
        line = '{}\t loss {:.4f}\tacc {:.3f}\tauc {:.3f}'.format(str(epoch), loss, accuracy, auc)
        for idx, acc_label in enumerate(acc_label_list):
            line += '\tlabel_{:d} {:d}%'.format(idx, int(acc_label))
        f.write(line + '\n')


class TabularDataset(Dataset):
    def __init__(self, data, cat_cols=None, output_col=None, identification_cols=None):
        """
        Characterizes a Dataset for PyTorch
        Parameters
        ----------
        data: pandas data frame
          The data frame object for the input data.
          It must contain all the continuous, categorical and the output columns to be used.
        cat_cols: List of strings
          The names of the categorical columns in the data.
          These columns will be passed through the embedding layers in the model.
          These columns must be label encoded beforehand.
        output_col: string
          The name of the output variable column in the data provided.
        """

        self.n = data.shape[0]

        if output_col:
            # self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
            self.y = data[output_col].astype(np.long).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns
                          if col not in self.cat_cols + [output_col] + identification_cols]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

        if identification_cols:
            self.identification = data[identification_cols].values
        else:
            self.identification = [None] * self.n

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        # print('idx', idx)
        # print('1', self.y[idx])
        # print('2', self.cont_X[idx])
        # print('3', self.cat_X[idx])
        # print(type(self.identification))
        # print(len(self.identification))
        #
        # print('4', self.identification[idx])
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx], self.identification[idx]]



def pred(output, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        return pred


def train_model(train_loader, model, criterion, optimizer, epoch, print_freq):
    train_loader = load_dataset(find_csv(args.data, 'train'))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (target, cont_input, cat_input, identification) in enumerate(train_loader):
        # measure data loading timepil_resize
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        target = target.squeeze()

        # compute output
        output = model(cont_input, cat_input)
        # loss = criterion(output, target)
        loss = criterion_smoothing(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), target.size(0))
        top1.update(prec1[0].cpu().data.numpy()[0], target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # nothing line
        gradClamp(model.parameters(), grad_clip)
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

    if args.evaluate:
        print('eval check')
        val_loader = load_dataset(find_csv(args.data, ''))
    else:
        val_loader = load_dataset(find_csv(args.data, 'val'))

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cnt_cnt_label = [0] * args.num_classes
    cnt_exact_pred = [0] * args.num_classes

    # switch to evaluate mode
    model.eval()

    if args.evaluate:
        evaluate_csv_file = os.path.join(args.result, 'evaluate.csv')
        feval = open(evaluate_csv_file, 'wt')

    with torch.no_grad():
        end = time.time()
        target_index_output, target_index_target = list(), list()

        for i, (target, cont_input, cat_input, identification) in enumerate(val_loader):

            target = target.cuda(non_blocking=True)
            target = target.squeeze()

            # compute output
            output = model(cont_input, cat_input)

            # loss = criterion(output, target)
            loss = criterion_smoothing(output, target)

            # --------------------------------------
            # for auroc get value from target index
            output_cpu = output.cpu().data.numpy()

            output_cpu = np.array([softmax(out)[args.target_index] for out in output_cpu])
            target_index_output.extend(output_cpu.astype(np.float))
            target_index_target.extend(np.equal(target.cpu().data.numpy(), args.target_index).astype(np.int))
            # --------------------------------------

            if args.evaluate:
                output_softmax = np.array([softmax(out) for out in output.cpu().numpy()])
                for ident, pred_values in zip(identification, output_softmax):
                    ident = str(np.squeeze(ident.data.numpy()))
                    idx_biger = np.argmax(pred_values)
                    diag = 'CKD' if idx_biger == 1 else 'AKI or NOR'

                    line = ','.join([ident, diag])
                    feval.write(line + '\n')

            # measure accuracy and record loss
            prec1 = accuracy(output, target, topk=(1,))

            losses.update(loss.item(), target.size(0))
            top1.update(prec1[0].cpu().data.numpy()[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # put together for acc per label
            pred_list = pred(output).cpu().numpy()[0]
            target_list = target.cpu().numpy()

            for (p, t) in zip(pred_list, target_list):
                cnt_cnt_label[t] += 1
                if p == t:
                    cnt_exact_pred[t] += 1

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

        log(' * Prec@1 {top1.avg:.3f} at Epoch {epoch:0}'.format(top1=top1, epoch=epoch))
        log(' * auc@1 {auc:.3f}'.format(auc=auc))

        acc_label_list = []
        for (i, (n_label, n_exact)) in enumerate(zip(cnt_cnt_label, cnt_exact_pred)):
            acc_label = (n_exact / n_label * 100) if n_label > 0 else 0
            acc_label_list.append(acc_label)
            log('acc of label {:0d}: {:0.3f}%'.format(i, acc_label))

    save_values(epoch, losses.avg, top1.avg, auc, acc_label_list)
    return auc


class FeedForwardNN(nn.Module):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts):

        """
        Parameters
        ----------
        emb_dims: List of two element tuples
            This list will contain a two element tuple for each categorical feature. The first element of a tuple will denote the number of unique values of the categorical feature.
            The second element will denote the embedding dimension to be used for that feature.
        no_of_cont: Integer
            The number of continuous features in the data.
        lin_layer_sizes: List of integers.
            The size of each linear layer. The length will be equal to the total number of linear layers in the network.
        output_size: Integer
            The size of the final output.
        emb_dropout: Float
            The dropout to be used after the embedding layers.
        lin_layer_dropouts: List of floats
            The dropouts to be used after each linear layer.
        """

        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont,
                                    lin_layer_sizes[0])

        self.lin_layers = \
            nn.ModuleList([first_lin_layer] + \
                          [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                           for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1],
                                      output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                        for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                            for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):

        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data

        for lin_layer, dropout_layer, bn_layer in \
                zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x


def load_dataset(path):
    data = pd.read_csv(path,
                       usecols=['AccNo', 'Diagnosis', 'KidneyLongCm', 'KidneyShortCm', 'Age', 'Sex', 'Height',
                                'Weight']).dropna()

    # categorical_features = ['Age', 'Sex', 'Height', 'Weight']
    categorical_features = ['Sex']
    output_feature = "Diagnosis"
    identification = ['AccNo']

    label_encoders = {}
    for cat_col in categorical_features:
        label_encoders[cat_col] = LabelEncoder()
        data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

    dataset = TabularDataset(data=data, cat_cols=categorical_features, output_col=output_feature, identification_cols=identification)

    batchsize = args.batch_size
    dataloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

    return dataloader


def get_dims(path):
    data = pd.read_csv(
        path,
        usecols=['AccNo', 'Diagnosis', 'KidneyLongCm', 'KidneyShortCm', 'Age', 'Sex', 'Height', 'Weight']).dropna()

    # categorical_features = ['Age', 'Sex', 'Height', 'Weight']
    categorical_features = ['Sex']

    label_encoders = {}
    for cat_col in categorical_features:
        label_encoders[cat_col] = LabelEncoder()
        data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

    cat_dims = [int(data[col].nunique()) for col in categorical_features]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    return cat_dims, emb_dims


if __name__ == '__main__':
    # # fix random seed of CPU and GPUs
    seed = 30
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cat_dims, emb_dims = [2], [(2, 1)] #get_dims(find_csv(args.data, 'train'))

    # create model
    model = FeedForwardNN(emb_dims, no_of_cont=5, lin_layer_sizes=[50, 100], output_size=2, emb_dropout=0.04,
                          lin_layer_dropouts=[0.2, 0.2])
    # lin_layer_dropouts=[0.001, 0.01])

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    transforms = transforms.Compose(transforms.ToTensor())
    dumy = lambda x: x

    # start main loop
    main(args, model, dummyImageLoader, dummyImageLoader, None, optimizer,
         train_dataset=DummyDataset, valid_dataset=DummyDataset,
         train_model=train_model, validate_model=validate_model,
         train_transforms=dumy, val_transforms=dumy)
