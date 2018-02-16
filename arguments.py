import numpy as np
import argparse
import time


def get_args():
    parser = argparse.ArgumentParser(description='sdca')

    parser.add_argument('--dataset', type=str, default='conll',
                        help='which dataset to use')
    parser.add_argument('--regularization', type=float, default=1.,
                        help='value of the l2 regularization parameter')
    parser.add_argument('--npass', type=int, default=100,
                        help='maximum number of pass over the trainset duality gaps used in the non-uniform sampling and to get a convergence criterion.')
    parser.add_argument('--sampling-scheme', type=str, default='gap',
                        help='Type of sampling: gap')
    parser.add_argument('--non-uniformity', type=float, default=0.8,
                        help='between 0 and 1. probability of sampling non-uniformly.')
    parser.add_argument('--sampler-period', type=int, default=None,
                        help='if not None, period to do a full batch update of the duality gaps, for the non-uniform sampling. Expressed as a number of epochs. This whole epoch will be counted in the number of pass used by sdca')
    parser.add_argument('--precision', type=float, default=1e-7,
                        help='Precision necessary to end the task.')
    parser.add_argument('--fixed-step-size', type=float, default=None,
                              help='if None, SDCA will use a line search. Otherwise should be a positive float to be used as the constant step size')
    parser.add_argument('--warm-start', type=np.array, default=None,
                        help='if numpy array, used as marginals to start from.')

    args = parser.parse_args()

    args.dense = True if args.dataset == 'ocr' else False

    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    args.dirname = "logs/" + time_stamp \
              + args.dataset \
              + args.sampling + "_" \
              + str(args.non_uniformity) + "_" \
              + str(args.sampler_period)

    return args