import argparse
import time

import numpy as np

import chunk.features
import ocr.features


def get_args():
    parser = argparse.ArgumentParser(description='sdca')

    parser.add_argument('--dataset', type=str, default='ocr',
                        help='which dataset to use')
    parser.add_argument('--train-size', type=int, default=100,
                        help='set to None if you want the full data set.')
    parser.add_argument('--test-size', type=int, default=50,
                        help='set to None if you want the full data set.')
    parser.add_argument('--regularization', type=float, default=1e-3,
                        help='value of the l2 regularization parameter')
    parser.add_argument('--npass', type=int, default=100,
                        help='maximum number of pass over the trainset duality gaps used in the '
                             'non-uniform sampling and to get a convergence criterion.')
    parser.add_argument('--sampling-scheme', type=str, default='gap',
                        help='Type of sampling. Options are "uniform" (default), "importance", '
                             '"gap", "gap+"')
    parser.add_argument('--non-uniformity', type=float, default=0.8,
                        help='between 0 and 1. probability of sampling non-uniformly.')
    parser.add_argument('--sampler-period', type=int, default=None,
                        help='if not None, period to do a full batch update of the duality gaps, '
                             'for the non-uniform sampling. Expressed as a number of epochs. '
                             'This whole epoch will be counted in the number of pass used by sdca')
    parser.add_argument('--precision', type=float, default=1e-7,
                        help='Precision wanted on the duality gap.')
    parser.add_argument('--fixed-step-size', type=float, default=None,
                        help='if None, SDCA will use a line search. Otherwise should be a '
                             'positive float to be used as the constant step size')
    parser.add_argument('--warm-start', type=np.array, default=None,
                        help='if numpy array, used as marginals to start from.')

    args = parser.parse_args()

    if args.dataset == 'ocr':
        args.data_train_path = 'data/ocr_train.mat'
        args.data_test_path = 'data/ocr_test.mat'
    elif args.dataset == 'conll':
        args.data_train_path = 'data/coNLL_train.mat'
        args.data_test_path = 'data/coNLL_test.mat'
    elif args.dataset == 'ner':
        args.data_train_path = 'data/NER_train.mat'
        args.data_test_path = 'data/NER_test.mat'
    elif args.dataset == 'pos':
        args.data_train_path = 'data/POS_train.mat'
        args.data_test_path = 'data/POS_test.mat'
    else:
        raise ValueError('the dataset is not defined')

    args.dense = True if args.dataset == 'ocr' else False

    args.features_cls = ocr.features if args.dataset == 'ocr' else chunk.features

    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    args.logdir = "logs/{}_n{}/{}_{}{}".format(
        args.dataset,
        'full' if (args.train_size is None) else str(args.train_size),
        time_stamp,
        args.sampling_scheme,
        str(args.non_uniformity)
    )

    return args
