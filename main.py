import os

from arguments import get_args
from get_datasets import get_datasets
from sdca import sdca

if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.logdir)

    # load datasets
    train_data, test_data = get_datasets(args)

    # launch optimization
    fullweights, fullmargs = sdca(trainset=train_data, testset=test_data, args=args)