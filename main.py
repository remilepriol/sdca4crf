import os

from sdca4crf.arguments import get_args
from sdca4crf.get_datasets import get_datasets
from sdca4crf.sdca import sdca

# TODO reproduce the results of Schmidt on conll

if __name__ == '__main__':
    args = get_args()

    # load datasets
    train_data, test_data = get_datasets(args)

    args.regularization = 1 / args.train_size

    infostring = (f"Data set: {args.dataset} \n"
                  f"Size of training set: {args.train_size} ({train_data.nb_points}) \n"
                  f"Size of test set: {args.test_size} ({test_data.nb_points}) \n"
                  f"Number of labels: {train_data.nb_labels} \n"
                  f"Number of features: {train_data.nb_features}")
    print(infostring)

    os.makedirs(args.logdir)
    with open(args.logdir + '/parameters.txt', 'w') as file:
        file.write(infostring)
        for arg in vars(args):
            file.write("{}:{}".format(arg, getattr(args, arg)))

    # run optimization
    fullweights, fullmargs = sdca(trainset=train_data, testset=test_data, args=args)

