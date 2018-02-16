from labeled_data import LabeledSequenceData
import chunk.features


def get_datasets(args):
    xtrain, ytrain, attributes = chunk.parse.read_mat(args.data_train_path, nb_sentences=train_size)
    xtest, ytest, _ = chunk.parse.read_mat(args.data_test_path, attributes=attributes, nb_sentences=test_size)
    trainset = LabeledSequenceData(xtrain, ytrain)
    testset = LabeledSequenceData(xtest, ytest)
    return trainset, testset


