import os
import sys
import time
# import glob
import numpy as np
from pyparsing import line
import torch
import logging
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from sparsemax import Sparsemax

from args_for_search import args, beta_decay_scheduler
import utils
from model_search import Network
from architect import Architect


def main():
    start_time = time.strftime("%Y%m%d-%H%M%S")
    args.save = 'logs/search-{}-{}'.format(args.save, start_time)
    utils.create_exp_dir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    writer = SummaryWriter(log_dir="./runs/{}".format(start_time))

    if args.cifar100:
        CIFAR_CLASSES = 100
    else:
        CIFAR_CLASSES = 10

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(precision=4, linewidth=120, sci_mode=False)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # prepare dataset
    if args.cifar100:
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    # build network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    # execution
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        if args.auxiliary_skip:
            beta_decay_scheduler.step(epoch)
            logging.info('skip beta decay rate %f', beta_decay_scheduler.decay_rate)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # sparsemax(α)のprint 
        logging.info(model.activation_func(model.alphas_normal))
        logging.info(model.activation_func(model.alphas_reduce))
        # 素のαのprint
        # print(model.alphas_normal)
        # print(model.alphas_reduce)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('search/accuracy/train', train_acc, epoch)
        writer.add_scalar('search/loss/train', train_obj, epoch)
        # writer.add_scalar('search/loss/arch', train_obj_arch, epoch)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('search/accuracy/valid', valid_acc, epoch)
        writer.add_scalar('search/loss/valid', valid_obj, epoch)

        # 学習率スケジューラの更新
        scheduler.step()

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
    """
        引数： {
            train_queue: trainデータのデータローダー,
            valid_queue: validデータのデータローダー,
            model: Networkクラス（ネットワークアーキテクチャの実体）,
            architect: Architectクラス（アーキテクチャ探索用のクラス）,
            criterion: cross entropy loss,
            optimizer: optimizer,
            lr: 学習率（スケジューラによって変化するため）,
            epoch: 現在のエポック,
        }
    """

    objs = utils.AvgrageMeter()  # network重み学習のloss
    # objs_arch = utils.AvgrageMeter()  # architecture重み学習のloss
    top1 = utils.AvgrageMeter()  # accuracy (top1)
    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # args.epochs*args.rate_epochs_to_changeエポックで，活性化関数をsoftmaxからsparsemaxに変更
        if (args.change_activation_func) and (epoch == int(args.epochs * args.rate_epochs_to_change)):
            model.activation_func = Sparsemax(dim=-1)

        if epoch >= 15:
            # 1バッチ分のデータ取得（アーキテクチャ探索に用いる用）
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)

            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            # アーキテクチャ(α)探索　（∂Lval(ω - lr * [∂Ltrain(ω,α) / ∂ω],α) / ∂α）
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        # 重み(ω)学習
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # loss, accuracy出力
        prec1 = utils.accuracy(logits, target)[0]
        objs.update(loss.item(), n)
        # objs_arch.update(loss_arch.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1 = utils.accuracy(logits, target)[0]
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
