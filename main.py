import os

from arguments import get_args
from get_datasets import get_datasets
from sdca import sdca

if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.logdir)

    # load datasets
    train_data, test_data = get_datasets(args)

    args.train_size = len(train_data)
    args.test_size = len(test_data)
    args.regularization = 1 / args.train_size

    # run optimization
    fullweights, fullmargs = sdca(trainset=train_data, testset=test_data, args=args)