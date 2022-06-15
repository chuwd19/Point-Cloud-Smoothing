
import os
import sys
sys.path.append('.')
sys.path.append('..')

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from tensorboardX import SummaryWriter
from datasets import get_dataset, DATASETS, get_num_classes
from semantic.core import SemanticSmooth
from time import time
import torch
import torchvision
import datetime
from architectures import get_architecture
from semantic.transformers import gen_transformer
from torch.utils.data import DataLoader
import math

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument('transtype', type=str, help='type of semantic transformations',
                    choices=['noise', 'points-rotation', 'points-shear', 'points-twist', 'points-taper', 'points-linear',
                    'points-noise', 'points-twist-rotationz'])
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--rotation_angle", type=float, default=0.0,
                    help="rotation_sd for twist-rotation composition")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--device', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--th", type=float, default=0, help="pre-defined radius for true robust counting")
parser.add_argument("--th_rotation", type=float, default=0, help="pre-defined radius for true robust counting")
parser.add_argument("--axis",type=str, default='z')
args = parser.parse_args()

if args.device is not None:
    device = args.device
else:
    device = 'cpu'

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    print('arch:', checkpoint['arch'])
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier = base_classifier.to(device)
    # prepare output file
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1,num_workers=1)

    canopy = None
    for data in dataloader:
        points,faces, label = data
        canopy = points[0]
        break
    transformer = gen_transformer(args, canopy)

    if args.transtype == 'points-twist-rotationz':
        args.th = math.sqrt((args.th / args.noise_sd) ** 2 + (args.th_rotation / args.rotation_angle) ** 2)
    # init tensorboard writer
    writer = SummaryWriter(os.path.dirname(args.outfile))

    # create the smooothed classifier g
    smoothed_classifier = SemanticSmooth(base_classifier, get_num_classes(args.dataset), transformer)

    tot_clean, tot_good, tot = 0, 0, 0

    for i, data in enumerate(dataloader):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        point, face, label = data
        label = torch.squeeze(label).to(device)
        point = point.float().to(device)
        before_time = time()
        # certify the prediction of g around x
        prediction, radius = smoothed_classifier.certify(point, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        # print(i, time_elapsed, correct, radius, file=f, flush=True)

        tot += 1
        tot_clean += correct
        tot_good += int(radius > args.th if correct > 0 else 0)
        writer.add_scalar('certify/clean_acc', tot_clean / tot, i)
        # writer.add_scalar('certify/robust_acc', tot_cert / tot, i)
        writer.add_scalar('certify/true_robust_acc', tot_good / tot, i)

    print(tot_clean / tot, tot_good/tot, file=f, flush=True)
    f.close()
