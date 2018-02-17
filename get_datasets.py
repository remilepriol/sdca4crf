from scipy.io import loadmat

from labeled_data import LabeledSequenceData


def read_mat(path):
    data = loadmat(path, squeeze_me=True)
    return data['X'], data['y']


def get_datasets(args):
    xtrain, ytrain = read_mat(args.data_train_path)
    trainset = LabeledSequenceData(xtrain, ytrain, args.train_size)

    xtest, ytest = read_mat(args.data_test_path)
    testset = LabeledSequenceData(xtest, ytest, args.test_size)

    return trainset, testset


