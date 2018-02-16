from labeled_data import LabeledSequenceData
import chunk.features
from scipy.io import loadmat


def read_mat(path):
    mat = loadmat(path, squeeze_me=True)
    xtrain = mat['X']
    ytrain = mat['y']
    return xtrain, ytrain


def get_datasets(args):
    xtrain, ytrain = read_mat(args.data_train_path)
    trainset = LabeledSequenceData(xtrain, ytrain)

    if args.data_test_path is not None:
        xtest, ytest = read_mat(args.data_test_path)
        testset = LabeledSequenceData(xtest, ytest)
    else:
        testset = None

    return trainset, testset


