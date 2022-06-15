import os
import sys
sys.path.append('.')
sys.path.append('..')

import math

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from semantic.core import SemanticSmooth
from time import time
import random
# import setproctitle
import torch
import torchvision
import datetime
import numpy as np
from tensorboardX import SummaryWriter

from architectures import get_architecture
from semantic.transformers import NoiseTransformer, PointTwistTaperRotationNoise
from torch.utils.data import DataLoader

EPS = 1e-6

parser = argparse.ArgumentParser(description='Strict rotation certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument("--l2_r", type=float, default=0.0, help="additional l2 magnitude to be tolerated")
# parser.add_argument("aliasfile", type=str, help='output of alias data')
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--th_twist", type=float,default = 0.0)
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=500)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--partial", type=float, default=180.0, help="certify +-partial degrees")
parser.add_argument("--verbstep", type=int, default=10, help="output frequency")
parser.add_argument("--th", type=float, default=0, help="")
parser.add_argument('--device', default=None, type=str,
                    help='id(s) for device')
args = parser.parse_args()
if args.device is not None:
    device = args.device
else:
    device = 'cpu'

args.partial = args.partial * math.pi /180
args.th_twist = args.th_twist * math.pi / 180
if __name__ == '__main__':
    orig_alpha = args.alpha
    args.alpha /= (args.slice ** 3 * args.th * args.partial * args.th_twist * 8)

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1,num_workers=1)
    canopy = None
    for data in dataloader:
        points,faces, label = data
        canopy = points[0]
        # print(points.shape,label.shape)
        break
    transformer = NoiseTransformer(args.noise_sd)
    twist_taper_rotation = PointTwistTaperRotationNoise(0,0,0,0,canopy)



    # prepare output file
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # init tensorboard writer
    writer = SummaryWriter(os.path.dirname(args.outfile))

    # create the smooothed classifier g
    smoothed_classifier = SemanticSmooth(base_classifier, get_num_classes(args.dataset), transformer)

    tot, tot_clean, tot_good, tot_cert = 0, 0, 0, 0

    # margin0 = args.partial/180 * math.pi**3 * math.sqrt(2)/ (2*args.slice)
    print(canopy.shape[0])
    margin = math.sqrt(canopy.shape[0] * (1 + 27/4 * (1+args.th) ** 2)) / args.slice
    print(margin)
    for i, data in enumerate(dataloader):

        if i < args.start:
            continue

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max >= 0:
            break

        point, face, label = data
        label = torch.squeeze(label).to(device)
        point = point.float().to(device)

        before_time = time()
        cAHat = smoothed_classifier.predict(point, args.N0, orig_alpha, args.batch)

        clean, cert, good = (cAHat == label), True, True
        # print(clean)
        gap = -1.0

        for j1 in range(int(args.slice*args.th)+1):
            if not good:
                break
            alpha = j1 /(2*args.slice) - args.th
            for j2 in range(int(args.slice*args.partial)+1):
                theta = j2 /(2*args.slice) - args.partial
                theta *= 180/math.pi
                for j3 in range(int(args.slice*args.th_twist)+1):
                    phi = j3 /(2*args.slice) - args.th_twist
                    phi *= 180/math.pi
                    now_x = twist_taper_rotation.rotation_adder.proc(point, theta).type_as(point).to(device)
                    now_x = twist_taper_rotation.taper_adder.raw_proc(now_x, alpha).type_as(point)
                    now_x = twist_taper_rotation.twist_adder.raw_proc(now_x, phi).type_as(point)
                    prediction, gap = smoothed_classifier.certify(now_x, args.N0, args.N, args.alpha, args.batch,
                                                            cAHat=cAHat, margin_sq=margin**2)
                    # print(f"> {j1, j2}/{args.slice ** 2} {str(datetime.timedelta(seconds=(time() - before_time)))}", end='\r', flush=True)
                    if prediction != cAHat or gap < 0 or cAHat == smoothed_classifier.ABSTAIN:
                        print(prediction)
                        print(cAHat)
                        print(gap)
                        print(f'not robust @ slice #{j1, j2}')
                        good = cert = False
                        break
                    elif prediction != label:
                        # the prediction is robustly wrong:
                        print(f'wrong @ slice #{j1, j2}')
                        # make gap always smaller than 0 for wrong slice
                        gap = - abs(gap) - 1.0
                        good = False
                        # robustly wrong is also skipped
                        # now "cert" is not recorded anymore
                        break
                    # else it is good


        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, cAHat, gap, clean, time_elapsed), file=f, flush=True)

        tot, tot_clean, tot_cert, tot_good = tot + 1, tot_clean + int(clean), tot_cert + int(cert), tot_good + int(good)
        print(f'{i} {gap >= 0.0} '
              f'CleanACC = {tot_clean}/{tot} = {float(tot_clean) / float(tot)} '
              # f'CertAcc = {tot_cert}/{tot} = {float(tot_cert) / float(tot)} '
              f'RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)} '
              f'Time = {time_elapsed}')

        writer.add_scalar('certify/clean_acc', tot_clean / tot, i)
        # writer.add_scalar('certify/robust_acc', tot_cert / tot, i)
        writer.add_scalar('certify/true_robust_acc', tot_good / tot, i)

    print(f'CleanACC = {tot_clean}/{tot} = {float(tot_clean) / float(tot)} '
        # f'CertAcc = {tot_cert}/{tot} = {float(tot_cert) / float(tot)} '
        f'RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)}', file=f, flush=True)

    f.close()



