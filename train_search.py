import os
import sys
import time
import random
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

from data.data import get_loaders, get_num_channels, get_num_classes
from torch.autograd import Variable
from micro_child import CNN
from micro_controller import Controller
from logger import Logger


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='root folder of the dataset')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name (cifar10, addnist, multnist, cifartile, language, gutenberg, geoclassing, chesseract, gameoflife)')
parser.add_argument('--num_classes', type=int, default=None, help='number of output classes (inferred from dataset if not set)')
parser.add_argument('--num_channels', type=int, default=None, help='number of input channels (inferred from dataset if not set)')
parser.add_argument('--batch_size', type=int, default=160, help='batch size')
parser.add_argument('--no-logger', action='store_true', help='disable experiment logging')
parser.add_argument('--no-augment', action='store_true', help='disable data augmentation')
parser.add_argument('--logger_api', type=str, default='wandb', help='logging backend (mlflow or wandb)')
parser.add_argument('--port', type=int, default=27027, help='logging server port')
parser.add_argument('--log_path', type=str, default=None, help='local path for logging storage')
parser.add_argument('--tmpdir', type=str, default=None, help='base directory for all saving and logging output')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--exp_name', type=str, default="NAS")

parser.add_argument('--child_lr_max', type=float, default=0.05)
parser.add_argument('--child_lr_min', type=float, default=0.0005)
parser.add_argument('--child_lr_T_0', type=int, default=10)
parser.add_argument('--child_lr_T_mul', type=int, default=2)
parser.add_argument('--child_num_layers', type=int, default=6)
parser.add_argument('--child_out_filters', type=int, default=20)
parser.add_argument('--child_num_branches', type=int, default=5)
parser.add_argument('--child_num_cells', type=int, default=5)
parser.add_argument('--child_use_aux_heads', type=bool, default=False)

parser.add_argument('--controller_lr', type=float, default=0.0035)
parser.add_argument('--controller_tanh_constant', type=float, default=1.10)
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)

parser.add_argument('--lstm_size', type=int, default=64)
parser.add_argument('--lstm_num_layers', type=int, default=1)
parser.add_argument('--lstm_keep_prob', type=float, default=0)
parser.add_argument('--temperature', type=float, default=5.0)

parser.add_argument('--entropy_weight', type=float, default=0.0001)
parser.add_argument('--bl_dec', type=float, default=0.99)

args = parser.parse_args()

_save_name = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = os.path.join(args.tmpdir, _save_name) if args.tmpdir else _save_name
if args.log_path is None and args.tmpdir is not None:
    args.log_path = args.tmpdir
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10

baseline = None
epoch = 0

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    train_loader, reward_loader, val_loader, test_loader = get_loaders(args)

    args.num_classes = get_num_classes(train_loader, args)
    args.num_channels = get_num_channels(train_loader, args)
    model = CNN(args)
    model.cuda()

    controller = Controller(args)
    controller.cuda()
    baseline = None

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    controller_optimizer = torch.optim.Adam(
        controller.parameters(),
        args.controller_lr,
        betas=(0.1,0.999),
        eps=1e-3,
    )

    scheduler = utils.LRScheduler(optimizer, args)

    logger = Logger(experiment_name=args.exp_name, port=args.port, api=args.logger_api, enabled=not args.no_logger)
    logger.setup_tracking(port=args.port, file_path=args.log_path)

    with logger(group="ENAS"):
        for attr, value in sorted(vars(args).items()):
            if attr in ("no_logger", "api", "exp_name", "port", "log_path", "tmpdir", "log_path"):
                continue
            logger.log_parameter(attr, str(value))

        for epoch in range(args.epochs):
            lr = scheduler.update(epoch)
            logging.info('epoch %d lr %e', epoch, lr)

            # training
            train_acc, train_loss = train(train_loader, model, controller, optimizer)
            logging.info('train_acc %f', train_acc)
            logger.log_metrics({
                'training/train accuracy': train_acc / 100.,
                'training/train loss': train_loss,
            }, step=epoch, step_name="epoch")

            train_controller(reward_loader, model, controller, controller_optimizer)

            # validation (reward split — used for controller decisions)
            val_acc, val_loss = infer(val_loader, model, controller)
            logging.info('valid_acc %f', val_acc)
            logger.log_metrics({
                'training/val accuracy': val_acc / 100.,
                'training/val loss': val_loss,
            }, step=epoch, step_name="epoch")

            # test (held-out set — no decisions made)
            test_acc, test_loss = infer(test_loader, model, controller)
            logging.info('test_acc %f', test_acc)
            logger.log_metrics({
                'training/test accuracy': test_acc / 100.,
                'training/test loss': test_loss,
            }, step=epoch, step_name="epoch")

            weights_path = os.path.join(args.save, 'weights.pt')
            utils.save(model, weights_path)

            with torch.no_grad():
                controller.eval()
                sample_dag, _, _ = controller()
            nb_params = model.count_active_params(*sample_dag)
            logger.log_metric('training/nb of parameters', nb_params, step=epoch, step_name="epoch")
            logging.info('nb_params %d', nb_params)

        logger.log_pytorch_model(model, f"ENAS_{args.dataset}", x=None, path=args.save, run_id=False)


def train(train_loader, model, controller, optimizer):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()

    for step, (data, target) in enumerate(train_loader):
        model.train()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        controller.eval()
        dag, _, _ = controller()

        logits, _ = model(data, dag)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, total_loss.avg, total_top1.avg)

    return total_top1.avg, total_loss.avg

def train_controller(reward_loader, model, controller, controller_optimizer):
    global baseline
    total_loss = utils.AvgrageMeter()
    total_reward = utils.AvgrageMeter()
    total_entropy = utils.AvgrageMeter()

    #for step, (data, target) in enumerate(reward_loader):
    for step in range(300):
        data, target = reward_loader.next_batch()
        model.eval()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        controller_optimizer.zero_grad()

        controller.train()
        dag, log_prob, entropy = controller()

        with torch.no_grad():
            logits, _ = model(data, dag)
            reward = utils.accuracy(logits, target)[0]

        if args.entropy_weight is not None:
            reward += args.entropy_weight*entropy

        log_prob = torch.sum(log_prob)
        if baseline is None:
            baseline = reward
        baseline -= (1 - args.bl_dec) * (baseline - reward)

        loss = log_prob * (reward - baseline)
        loss = loss.sum()

        loss.backward()

        controller_optimizer.step()

        total_loss.update(loss.item(), n)
        total_reward.update(reward.item(), n)
        total_entropy.update(entropy.item(), n)

        if step % args.report_freq == 0:
            #logging.info('controller %03d %e %f %f', step, loss.item(), reward.item(), baseline.item())
            logging.info('controller %03d %e %f %f', step, total_loss.avg, total_reward.avg, baseline.item())
            #tensorboard.add_scalar('controller/loss', loss, epoch)
            #tensorboard.add_scalar('controller/reward', reward, epoch)
            #tensorboard.add_scalar('controller/entropy', entropy, epoch)

def infer(valid_loader, model, controller):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.eval()
    controller.eval()

    with torch.no_grad():
        for step, (data, target) in enumerate(valid_loader):
            data = data.cuda()
            target = target.cuda()

            dag, _, _ = controller()

            logits, _ = model(data, dag)
            loss = F.cross_entropy(logits, target)

            prec1 = utils.accuracy(logits, target)[0]
            n = data.size(0)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, loss.item(), prec1.item())
                logging.info('normal cell %s', str(dag[0]))
                logging.info('reduce cell %s', str(dag[1]))

    return total_top1.avg, total_loss.avg


if __name__ == '__main__':
    main() 

