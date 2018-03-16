import numpy as np

from sdca4crf.arguments import get_args, init_logdir
from sdca4crf.get_datasets import get_datasets
from sdca4crf.sdca import sdca

if __name__ == '__main__':
    args = get_args()

    # load datasets
    train_data, test_data = get_datasets(args)

    if args.regularization is None:
        args.regularization = 1 / args.train_size

    init_logdir(args)

    # run optimization
    optweights, optmargs = sdca(trainset=train_data, testset=test_data, args=args)

    print("Optimization finished.")
    if args.save == 'all':
        np.save(args.logdir + '/optweights.npy', optweights.to_array())
        marginals_dic = {'marginal' + str(i): margs.binary for i, margs in enumerate(optmargs)}
        np.savez_compressed(args.logdir + '/optmarginals.npy', **marginals_dic)
