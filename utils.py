import functools
import math
import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def singleton(cls, *args, **kw):
    instances = dict()
    @functools.wraps(cls)
    def _fun(*clsargs, **clskw):
        if cls not in instances:
            instances[cls] = cls(*clsargs, **clskw)
        return instances[cls]
    _fun.cls = cls  # make sure cls can be obtained
    return _fun


@singleton
class DecayScheduler(object):
    def __init__(self, base_lr=1.0, T_max=50, T_start=0, T_stop=50, decay_type='cosine'):
        self.base_lr = base_lr
        self.T_max = T_max
        self.T_start = T_start
        self.T_stop = T_stop        
        self.decay_type = decay_type
        self.decay_rate = 1.0

    def step(self, epoch):
        if epoch >= self.T_start:
            if self.decay_type == "cosine":
                self.decay_rate = self.base_lr * (1 + math.cos(math.pi * epoch / (self.T_max - self.T_start))) / 2.0 if epoch <= self.T_stop else self.decay_rate
            elif self.decay_type == "slow_cosine":
                self.decay_rate = self.base_lr * math.cos((math.pi/2) * epoch / (self.T_max - self.T_start)) if epoch <= self.T_stop else self.decay_rate
            elif self.decay_type == "linear":
                self.decay_rate = self.base_lr * (self.T_max - epoch) / (self.T_max - self.T_start) if epoch <= self.T_stop else self.decay_rate
            else:
                self.decay_rate = self.base_lr
        else:
            self.decay_rate = self.base_lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    # 分類結果のindexをpredに代入。sizeは，(batch_size, maxk)
    _, pred = output.topk(maxk, 1, True, True)
    # 転置により，[[top1のリスト], [top2のリスト], ...]　を得る
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

# 確率の応じて，バッチ毎にエッジごと切る
def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob

        # mask: keep_probの確率で，中身が０または１のテンソル(x.size(0), 1, 1, 1)
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
