import os
import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import torch
import torchvision
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from tensorboardX import SummaryWriter
from train_utils import AverageMeter, accuracy, init_logfile, log
from semantic.transformers import gen_transformer, AbstractTransformer

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('transtype', type=str, help='type of semantic transformations',
                    choices=['rotation-noise','points-rotation', 'points-shear', 'points-twist',
                    'points-taper', 'points-taper-noise', 'points-rotation-noise', 'points-twist-rotationz',
                    'points-taper-rotationz', 'points-linear', 'points-twist-taper-rotationz', 'points-noise',
                    'noise', 'rotation', 'strict-rotation-noise', 'translation', 
                    'brightness', 'resize', 'gaussian', 'btranslation', 'expgaussian', 'foldgaussian',
                    'rotation-brightness', 'rotation-brightness-contrast', 'resize-brightness',
                    'universal'])
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--rotation_angle', help='constrain the rotation angle to +- rotation angle in degree',
                    type=float, default=180.0)
parser.add_argument('--taper_angle', help='constrain the taper parameter to +- taper_angle',
                    type=float, default=0.1)
parser.add_argument('--twist_angle', help='constrain the twist angle to +- twist_angle in degree',
                    type=float, default=0)
parser.add_argument('--device', default=None, type=str,
                    help='id(s) for to(device)')
parser.add_argument('--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrain', default=None, type=str)
##################### arguments for consistency training #####################
parser.add_argument('--num_noise_vec', default=1, type=int,
                    help="number of noise vectors. `m` in the paper.")
parser.add_argument('--lbd', default=20., type=float)
##################### arguments for tensorboard print #####################
parser.add_argument('--print_step', action="store_true")
parser.add_argument('--axis', default='z', type=str, help="Rotation axis for 3D points")
args = parser.parse_args()

if args.device is not None:
    device = args.device
else:
    device = 'cpu'

def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), targets,
                    reduction=reduction)


def _cross_entropy(input, targets, reduction='mean'):
    targets_prob = F.softmax(targets, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()


def _entropy(input, reduction='mean'):
    return _cross_entropy(input, input, reduction)

def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers)

    model = get_architecture(args.arch, args.dataset).to(device)

    if args.pretrain is not None:
        if args.pretrain == 'pointnet':
            model.load_state_dict(torch.load("models/64p_natural.pth"))
            print('loaded from models/64p_natural')
        elif args.pretrain == 'pointnet1024':
            model.load_state_dict(torch.load("models/1024p_natural.pth"))
            print('loaded from models/1024p_natural')
        else:
            # load the base classifier
            checkpoint = torch.load(args.pretrain)
            model.load_state_dict(checkpoint['state_dict'])
            print(f'loaded from {args.pretrain}')

    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
    writer = SummaryWriter(args.outdir)

    canopy = None
    for data in train_loader:
        points, label = data
        canopy = points[0]
        break
    transformer = gen_transformer(args, canopy)

    criterion = CrossEntropyLoss().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    for epoch in range(args.epochs):
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, transformer, writer)
        test_loss, test_acc = test(test_loader, model, criterion, epoch, transformer, writer)
        after = time.time()

        scheduler.step(epoch)

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}%\t{:.3}\t{:.3}%".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc*100, test_loss, test_acc*100))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

def _chunk_minibatch(batch, num_batches):
    points, _  = batch
    batch_size = points.shape[0] // num_batches
    for i in range(num_batches):
        yield batch[i*batch_size : (i+1)*batch_size]



def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int,
          transformer: AbstractTransformer, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_reg = AverageMeter()
    confidence = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    train_correct = 0.0
    train_amount = 0.0
    # switch to train mode
    model.train()

    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for data in mini_batches:
            points, targets = data
            targets = torch.squeeze(targets).to(device)
            points = points.float().to(device)
            # targets = targets.to(device)
            batch_size = points.shape[0]

            noised_inputs = [transformer.process(points).to(device) for _ in range(args.num_noise_vec)]

            # augment inputs with noise
            inputs_c = torch.cat(noised_inputs, dim=0)
            # print(inputs_c.shape, targets.shape)
            targets_c = targets

            logits = model(inputs_c)

            loss_xent = criterion(logits, targets_c)
            # print(logits.shape)
            logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
            softmax = [F.softmax(logit, dim=1) for logit in logits_chunk]
            avg_softmax = sum(softmax) / args.num_noise_vec

            consistency = [kl_div(logit, avg_softmax, reduction='none').sum(1)
                           + _entropy(avg_softmax, reduction='none')
                           for logit in logits_chunk]
            consistency = sum(consistency) / args.num_noise_vec
            consistency = consistency.mean()

            loss = loss_xent + args.lbd * consistency
            # loss = loss_xent

            avg_confidence = -F.nll_loss(avg_softmax, targets)
            
            max_predict = logits.data.max(1)[1]
            # print(targets_c)
            correct = max_predict.eq(targets_c.data).float().mean(1).cpu().sum()
            train_correct += correct.item()
            train_amount += points.size()[0]
            # acc1, acc5 = accuracy(logits, targets_c, topk=(1, 5))
            losses.update(loss_xent.item(), batch_size)
            losses_reg.update(consistency.item(), batch_size)
            confidence.update(avg_confidence.item(), batch_size)
            # top1.update(acc1.item(), batch_size)
            # top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Accuracy {acc:.3f}%\t'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=100*train_correct/train_amount))

            if args.print_step:
                writer.add_scalar(f'epoch/{epoch}/loss/train', losses.avg, i)
                writer.add_scalar(f'epoch/{epoch}/loss/consistency', losses_reg.avg, i)
                writer.add_scalar(f'epoch/{epoch}/loss/avg_confidence', confidence.avg, i)
                writer.add_scalar(f'epoch/{epoch}/batch_time', batch_time.avg, i)
                # writer.add_scalar(f'epoch/{epoch}/accuracy/train@1', top1.avg, i)
                # writer.add_scalar(f'epoch/{epoch}/accuracy/train@5', top5.avg, i)

    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('loss/consistency', losses_reg.avg, epoch)
    writer.add_scalar('loss/avg_confidence', confidence.avg, epoch)
    writer.add_scalar('batch_time', batch_time.avg, epoch)
    # writer.add_scalar('accuracy/train@1', top1.avg, epoch)
    # writer.add_scalar('accuracy/train@5', top5.avg, epoch)

    return (losses.avg, train_correct/train_amount)


def test(loader, model, criterion, epoch, transformer: AbstractTransformer, writer=None, print_freq=25):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    test_correct = 0.0
    test_amount = 0.0
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)
            points, targets = data
            targets = torch.squeeze(targets).to(device)
            points = points.float().to(device)
            # inputs = inputs
            # targets = targets.to(device)()

            # augment inputs with noise
            inputs = transformer.process(points).to(device)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            max_predict = outputs.data.max(1)[1]
            correct = max_predict.eq(targets.data).float().mean(1).cpu().sum()
            test_correct += correct.item()
            test_amount += points.size()[0]
            # acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            # top1.update(acc1.item(), inputs.size(0))
            # top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Accuracy {acc:.3f}%\t'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, acc=100*test_correct/test_amount))

        if writer:
            writer.add_scalar('loss/test', losses.avg, epoch)
            # writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            # writer.add_scalar('accuracy/test@5', top5.avg, epoch)

        return (losses.avg, test_correct/test_amount)


if __name__ == "__main__":
    main()
