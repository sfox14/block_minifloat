"""
Code modified from Qpytorch repository. https://github.com/Tiiiger/QPyTorch/blob/master
"""

import argparse
import time
import torch
import torch.nn.functional as F
import utils
import tabulate
import bisect
import os
import sys
from functools import partial
import collections
import models
from data import get_data
#from qtorch.optim import OptimLP
from optim import OptimLP
from torch.optim import SGD
from quant import *
#from qtorch import FloatingPoint

num_types = ["weight", "activate", "error", "acc", "grad", "momentum"]

parser = argparse.ArgumentParser(description='Block Minifloat SGD training')
parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10','CIFAR100','IMAGENET'],
                    help='dataset name: CIFAR10, CIFAR100 or IMAGENET')
parser.add_argument('--data_path', type=str, default="/opt/datasets/", required=True, metavar='PATH',
                    help='path to datasets location (default: "./data")')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--model', type=str, default="ResNet18LP", required=True, metavar='MODEL',
                    help='model name (default: ResNet18LP)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N',
                    help='evaluation frequency (default: 5)')
parser.add_argument('--save_freq', type=int, default=15, metavar='N',
                    help='save checkpoint frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=200, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--tile', type=int, default=48, choices=[16, 24, 32, 48, -1, 0],
                    help='tile size for shared exponent (default: -1); -1 if image tensor')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--flush_to_zero', action='store_true', default=False,
                    help='use qtorch floating point quantizer which does not handle denormal numbers')
parser.add_argument('--gpu', type=str, help='comma separated list of GPU(s) to use')
parser.add_argument('--output_dir', type=str, default='./checkpoints', help='output directory')

for num in num_types:
    parser.add_argument('--{}-man'.format(num), type=int, default=-1, metavar='N',
                        help='number of bits to use for mantissa of {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-exp'.format(num), type=int, default=-1, metavar='N',
                        help='number of bits to use for exponent of {}; -1 if full precision.'.format(num))
    parser.add_argument('--{}-rounding'.format(num), type=str, default='stochastic', metavar='S',
                        choices=["stochastic","nearest"],
                        help='rounding method for {}, stochastic or nearest'.format(num))

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)

# gpu to use
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Setup log directory
time_str = time.strftime("%m_%d_%H_%M")
iden = "{}_{}_w{}{}_a{}{}_e{}{}".format(args.model, args.dataset, 
    args.weight_man, args.weight_exp, 
    args.activate_man, args.activate_exp,
    args.error_man, args.error_exp)
dir_name = os.path.join(args.output_dir, iden)

log_name = os.path.join(dir_name, "main_{}.log".format(time_str))
print('Checkpoint directory {}'.format(dir_name))
os.makedirs(dir_name, exist_ok=True)
with open(log_name, 'w') as f: f.write('python {}\n\n'.format(sys.argv))

# load dataset
loaders = get_data(args.dataset, args.data_path, args.batch_size)

# define quantizers
number_dict = {}
for i, num in enumerate(num_types):
    num_rounding = getattr(args, "{}_rounding".format(num))
    num_man = getattr(args, "{}_man".format(num))
    num_exp = getattr(args, "{}_exp".format(num))
    number_dict[num] = BlockMinifloat(exp=num_exp, man=num_man, 
            tile=args.tile, flush_to_zero=args.flush_to_zero)
    print("{:10}: {}".format(num, number_dict[num]))

weight_quantizer = quantizer(forward_number=number_dict["weight"],
                             forward_rounding=args.weight_rounding)
grad_quantizer   = quantizer(forward_number=number_dict["grad"],
                             forward_rounding=args.grad_rounding)
momentum_quantizer = quantizer(forward_number=number_dict["momentum"],
                               forward_rounding=args.momentum_rounding)
acc_quantizer = quantizer(forward_number=number_dict["acc"],
                               forward_rounding=args.acc_rounding)
acc_err_quant = lambda : Quantizer(number_dict["activate"], number_dict["error"],
                                     args.activate_rounding, args.error_rounding)

# Build model
print('Model: {}'.format(args.model))
model_cfg = getattr(models, args.model)
model_cfg.kwargs.update({"quant":acc_err_quant})

if args.dataset=="CIFAR10": num_classes=10
elif args.dataset=="CIFAR100": num_classes=100
elif args.dataset=="IMAGENET": num_classes=1000

if (args.model == "ResNet18LP") or (args.model == "MobileNetV2LP"): 
    model_cfg.kwargs.update({"image_size":224 if args.dataset=="IMAGENET" else 32})
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

# learning rate schedules
def default_schedule(epoch):
    t = (epoch) / args.epochs
    lr_ratio = 0.01
    t_const = 0.2 # 0.5 default
    if t <= t_const:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - t_const) / (0.9 - t_const)
    else:
        factor = lr_ratio
    return args.lr_init * factor

# https://github.com/uoguelph-mlrg/Cutout (200 epochs)
def cifar_schedule(epoch): 
    milestones = [60, 120, 160]
    gamma = 0.2
    lr_init = 0.05
    return lr_init * (gamma ** bisect.bisect_right(milestones, epoch) ) 

# ptorch examples (90 epochs)
def imagenet_schedule(epoch):
    lr_init = args.lr_init
    lr = lr_init * (0.1 ** (epoch // 30))
    return lr


if (args.dataset == "CIFAR10" or args.dataset == "CIFAR100") and args.epochs >= 200:
    schedule = lambda x: cifar_schedule(x)
elif args.dataset == "IMAGENET":
    schedule = lambda x: imagenet_schedule(x)
else:
    schedule = lambda x: default_schedule(x)


criterion = F.cross_entropy
optimizer = SGD( model.parameters(),
                 lr=args.lr_init,
                 momentum=0.9,
                 weight_decay=args.wd)

# resume training
start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    resume_keys = list(checkpoint['state_dict'])
    model_keys = list(model.state_dict())
    matched_state_dict = {
        model_keys[i]:checkpoint['state_dict'][k] for i,k in enumerate(resume_keys)}
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(matched_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])


optimizer = OptimLP(optimizer,
                    weight_quant=weight_quantizer,
                    grad_quant=grad_quantizer,
                    momentum_quant=momentum_quantizer,
                    acc_quant=acc_quantizer)

# Prepare logging
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_time', 'te_loss', 'te_acc', 'te_time']

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    train_res = utils.run_epoch(loaders['train'], model, criterion,
                                optimizer=optimizer, phase="train" )
    time_pass = time.time() - time_ep
    train_res['time_pass'] = time_pass

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        time_ep = time.time()
        
        utils.bn_update(loaders['train'], model)
        test_res = utils.run_epoch(loaders['test'], model, criterion, phase="eval") 

        time_pass = time.time() - time_ep
        test_res['time_pass'] = time_pass
    else:
        test_res = {'loss': None, 'accuracy': None, 'time_pass': None}

    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], train_res['time_pass'], 
                test_res['loss'], test_res['accuracy'], test_res['time_pass']]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
    with open(log_name, 'a') as f: f.write(table+'\n')

    # save checkpoint by epoch
    if epoch % args.save_freq == args.save_freq - 1 or epoch == args.epochs - 1:
        base_dir = os.path.join(dir_name, "base")
        os.makedirs(base_dir, exist_ok=True)
        utils.save_checkpoint(
            base_dir,
            epoch+1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )

    # save the most recent checpoint every epoch
    utils.save_checkpoint(
        dir_name,
        epoch+1,
        recent=True,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )



