from sdca import sdca
from arguments import get_args


if __name__ == '__main__':
    args = get_args()

    # load datasets
    feature_cls, train_data, test_data = get_features(args)

    # launch optimization
    sdca(features_cls=feature_cls, trainset=train_data, testset=test_data)