import svmlight
from get_folds import get_n_folds
from aux_funcs import merge_all_folds_except, get_neg_proportion
import config


def get_features(data):
    feature_count = count_features([item for file,_ in data for item in file])
    current_feature = 0
    feature_mapping = {}
    for k,v in feature_count.items():
        if v > config.MINIMUM_APPEARANCE_OF_FEATURE_IN_TRAINING:
            feature_mapping[k] = current_feature
            current_feature += 1
    return feature_mapping


def count_features(data):
    feature_count = {}
    for feature in data:
        if feature in feature_count.keys():
            feature_count[feature] = feature_count[feature] + 1
        else:
            feature_count[feature] = 1
    return feature_count


def prepare_data_for_svm(data, feature_mapping):
    prepared_data = []
    for file,tag in data:
        feature_count = count_features(file)
        list_features = []
        for feature, count in feature_count.items():
            if feature in features_mapping.keys():
                list_features.append((feature_mapping[feature], count))
        prepared_data.append((1 if tag == config.POSITVIE_TAG else -1, list_features))
    return prepared_data

folds = get_n_folds(10)
for i,test_fold in enumerate(folds):
    print("Fold: " + str(i))
    training = merge_all_folds_except(i, folds)
    features_mapping = get_features(training)

    svm_train = prepare_data_for_svm(training, features_mapping)
    print("Train Ratio = " + str(get_neg_proportion(training)))
    print("Test Ratio = " + str(get_neg_proportion(test_fold)))
    print(len(training))
    print(len(test_fold))

    model = svmlight.learn(svm_train, type='classification')

    svm_test = prepare_data_for_svm(test_fold, features_mapping)
    svm_test_with_unknown_class = [(0,features) for _,features in svm_test]
    # predictions = svmlight.classify(model, svm_test_with_unknown_class)
    pos_count = 0
    # for i,p in enumerate(predictions):
    #     truth = svm_test[i][0]
    #     if truth*p > 0:
    #         print("Correct: %.8f" % p)
    #         pos_count += 1
    #     else :
    #         print("Incorre: %.8f" % p)
    # print(pos_count)

# model data can be stored in the same format SVM-Light uses, for
# interoperability with the binaries.
# svmlight.write_model(model, 'my_model.dat')

# classify the test data. this function returns a list of numbers, which
# represent the classifications.
# predictions = svmlight.classify(model, test_data)
# for p in predictions:
#     print('%.8f' % p)
