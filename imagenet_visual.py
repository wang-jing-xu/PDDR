"""
ImageNet training script.
Including APEX (distributed training), and DALI(data pre-processing using CPU+GPU) provided by NIVIDIA.
Thanks pytorch demo, implus (Xiang Li from NJUST), DALI.
Author: Xu Ma
Date: Aug/15/2019
Email: xuma@my.unt.edu

Useage:
python3 -m torch.distributed.launch --nproc_per_node=8 main -a old_resnet50 --fp16 --b 32



"""
import logging
import sys
import argparse
import os
import sys
sys.path.append(os.getcwd())
import shutil
import time
import traceback
import warnings

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import models.Deploy as models
from utils import Logger, mkdir_p

from PIL import Image
import numpy as np

# Ignoring warnings
warnings.filterwarnings('ignore')

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    # import nvidia.dali.types as types

    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='/home/wjx/cv/dataset/leafimages', type=str)
parser.add_argument('-n', '--num-class', default=39, type=int)
parser.add_argument('--arch', '-a', metavar='ARCH', default='old_resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
# parser.add_argument('--arch', '-a', default='old_resnet18', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=30, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true', default=False,
                    help='Run model fp16 mode.')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                         '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')

parser.add_argument("--local_rank", default=0, type=int)

cudnn.benchmark = True


# class HybridTrainPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
#         super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size,
#                                     random_shuffle=True)
#
#         # let user decide which pipeline works him bets for RN version he runs
#         dali_device = 'cpu' if dali_cpu else 'gpu'
#         decoder_device = 'cpu' if dali_cpu else 'mixed'
#         # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
#         # without additional reallocations
#         device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
#         host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
#         self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
#                                                  # device_memory_padding=device_memory_padding,
#                                                  # host_memory_padding=host_memory_padding,
#                                                  random_aspect_ratio=[0.8, 1.25],
#                                                  random_area=[0.1, 1.0],
#                                                  num_attempts=100)
#         self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             crop=(crop, crop),
#                                             # image_type=types.RGB,
#                                             mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
#         self.coin = ops.CoinFlip(probability=0.5)
#         print('DALI "{0}" variant'.format(dali_device))
#
#     def define_graph(self):
#         rng = self.coin()
#         self.jpegs, self.labels = self.input(name="Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         output = self.cmnp(images.gpu(), mirror=rng)
#         return [output, self.labels]


#
# class HybridValPipe(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
#         super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
#         self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size,
#                                     random_shuffle=False)
#         self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
#         self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
#         self.cmnp = ops.CropMirrorNormalize(device="gpu",
#                                             dtype=types.FLOAT,
#                                             output_layout=types.NCHW,
#                                             crop=(crop, crop),
#                                             # image_type=types.RGB,
#                                             mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                                             std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
#
#     def define_graph(self):
#         self.jpegs, self.labels = self.input(name="Reader")
#         images = self.decode(self.jpegs)
#         images = self.res(images)
#         output = self.cmnp(images)
#         return [output, self.labels]


@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline_train(data_dir, crop):
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=True, name="Reader")
    images = fn.decoders.image_random_crop(images, device="mixed", output_type=types.RGB,
                                           random_aspect_ratio=[0.8, 1.25],
                                           random_area=[0.1, 1.0],
                                           num_attempts=100)
    images = fn.resize(images, device='gpu', resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
    rng = fn.random.coin_flip(probability=0.5)
    images = fn.crop_mirror_normalize(images.gpu(), mirror=rng, device='gpu', dtype=types.FLOAT,
                                      output_layout=types.NCHW,
                                      crop=(crop, crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return images, labels


@pipeline_def(num_threads=4, device_id=0)
def get_dali_pipeline_val(data_dir, crop, size):
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=False, name="Reader")
    images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    images = fn.resize(images, device='gpu', resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
    # rng = fn.random.coin_flip(probability=0.5)
    images = fn.crop_mirror_normalize(images, device="gpu", dtype=types.FLOAT, output_layout=types.NCHW,
                                      crop=(crop, crop),
                                      mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    return images, labels


# class GetDaliPipelineTrain(pipeline_def(num_threads=4, device_id=0)):
#     def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
#         super(GetDaliPipelineTrain, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)


best_prec1 = 0
args = parser.parse_args()

# checkpoint

seed = np.random.randint(1, 10000)

path = args.data
# 使用split函数将路径字符串根据'/'分割成一个列表
folders = path.split('/')
# 获取列表中最后一个非空字符串，即最后一个文件夹的名字
data_name = [folder for folder in folders if folder][-1]

if args.checkpoint is None:
    if args.fp16:
        args.checkpoint = 'checkpoints/' + data_name + '/' + args.arch + '_FP16_' + str(seed)
    else:
        args.checkpoint = 'checkpoints/' + data_name + '/' + args.arch + '_FP32_' + str(seed)

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

# make apex optional
if args.fp16 or args.distributed:
    print("Import APEX!")
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

screen_logger = logging.getLogger()
screen_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
screen_logger.addHandler(file_handler)


def printf(str):
    screen_logger.info(str)
    print(str)


def main():
    global best_prec1, args

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    if not os.path.isdir(args.checkpoint) and args.local_rank == 0:
        mkdir_p(args.checkpoint)

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    screen_logger = logging.getLogger()
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            printf("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    # create model
    if args.pretrained:
        printf("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        printf("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=args.num_class)

    printf(f"args: {args}")
    model = model.cuda()
    if args.fp16:
        model = network_to_half(model)
    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   verbose=False)

    # optionally resume from a checkpoint
    title = 'ImageNet-' + args.arch
    if args.resume:
        if os.path.isfile(args.resume):
            printf("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            printf("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.resume, checkpoint['epoch']))
            if args.local_rank == 0:
                logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            printf("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.local_rank == 0:
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Valid Top5.'])

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if (args.arch == "inception_v3"):
        crop_size = 299
        val_size = 320  # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    # pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank,
    #                        data_dir=traindir, crop=crop_size, dali_cpu=args.dali_cpu)
    # pipes = get_dali_pipeline_train(data_dir=traindir, crop=crop_size)
    # pipe.build()
    # train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    train_loaderr = DALIGenericIterator(
        [get_dali_pipeline_train(data_dir=traindir, batch_size=args.batch_size, crop=crop_size)],
        ['data', 'label'],
        reader_name='Reader'
    )
    # printf(train_loaderr)
    # save_dir = "image_output"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # 创建 DALI 迭代
    #
    # # 遍历迭代器并保存图像
    # for i, batch in enumerate(train_loaderr):
    #     printf("i :", i)
    #     images = batch[0]['data']
    #     labels = batch[0]['label']
    #
    #     # 将图像数据从 GPU 转移到 CPU，并转换为 numpy 数组
    #     image_np = images.cpu().numpy()
    #
    #     # 对每张图片进行保存
    #     for j in range(image_np.shape[0]):
    #         # 转换数据类型和范围
    #         image_np_uint8 = (image_np[j].transpose(1, 2, 0) * 255).astype(np.uint8)
    #
    #         # 创建图像对象
    #         image = Image.fromarray(image_np_uint8)
    #
    #         # 保存图像
    #         image_path = os.path.join(save_dir, f"image_{i * args.batch_size + j}.jpg")
    #         image.save(image_path)
    #
    #     # 假设只保存前 10 张图片
    #     # sys.exit()
    #     # if i >= 4:
    #     #     break
    #
    #
    # sys.exit()
    # =-=-=-=-=-=-=-=-=-=-=
    # pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank,
    #                      data_dir=valdir, crop=crop_size, size=val_size)
    # pipe.build()
    # val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    val_loaderr = DALIGenericIterator(
        [get_dali_pipeline_val(data_dir=valdir, batch_size=args.batch_size, crop=crop_size, size=320)],
        ['data', 'label'],
        reader_name='Reader'
    )

    if args.evaluate:
        validate(val_loaderr, model, criterion)
        return

    total_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        adjust_learning_rate(optimizer, epoch, args)

        if args.local_rank == 0:
            printf('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        [train_loss, train_acc, avg_train_time] = train(train_loaderr, model, criterion, optimizer, epoch)
        total_time.update(avg_train_time)
        # evaluate on validation set
        [test_loss, prec1, prec5] = validate(val_loaderr, model, criterion)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            # append logger file
            logger.append([optimizer.param_groups[0]['lr'], train_loss, test_loss, train_acc, prec1, prec5])

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint, filename="epoch" + str(epoch + 1))
            if epoch == args.epochs - 1:
                printf('##Top-1 {0}\n'
                       '##Top-5* {1}\n'
                       '##Perf  {2}'.format(best_prec1, prec5, args.total_batch_size / total_time.avg))

        # reset DALI iterators
        train_loaderr.reset()
        val_loaderr.reset()

    if args.local_rank == 0:
        logger.close()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        train_loader_len = int(train_loader._size / args.batch_size)

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i > 1:
            # if args.local_rank == 0:
            printf('[{0}/{1}]\t'
                   'Batch Time {batch_time.avg:.3f}\t'
                   'Data Time {data_time.avg:.3f}\t'
                   'Speed {2:.3f}\t'
                   'Loss {loss.avg:.4f}\t'
                   'Top1 {top1.avg:.3f}\t'
                   'Top5 {top5.avg:.3f}'.format(
                i, train_loader_len, args.total_batch_size / batch_time.avg,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5))
    return [losses.avg, top1.avg, batch_time.avg]


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        val_loader_len = int(val_loader._size / args.batch_size)

        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            # if args.local_rank == 0:
            printf('Test: [{0}/{1}]\t'
                   'Time {batch_time.avg:.3f}\t'
                   'Speed {2:.3f} \t'
                   'Loss {loss.avg:.4f}\t'
                   'Top1 {top1.avg:.3f}\t'
                   'Top5 {top5.avg:.3f}'.format(
                i, val_loader_len,
                args.total_batch_size / batch_time.avg,
                batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    if args.local_rank == 0:
        printf(' TEST Top1 {top1.avg:.4f} Top5 {top5.avg:.4f}'.format(top1=top1, top5=top5))

    return [losses.avg, top1.avg, top5.avg]


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if is_best:
        filepath = os.path.join(checkpoint, 'model_best.pth.tar')
        torch.save(state, filepath)
        # shutil.copyfile(filepath, os.path.join(checkpoint, filename + 'model_best.pth.tar'))


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """Comes from pytorch demo"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    output1 = output.t()
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target = target.view(1, -1).expand_as(pred)
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k]
        correct_k = correct_k.reshape(-1)
        correct_k = correct_k.float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        # os.system("sudo poweroff")
    print("DONE, FINISHED!!!")
    # os.system("sudo poweroff")
    # main()
