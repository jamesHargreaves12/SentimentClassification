import config


def merge_all_folds_except(index, folds):
    using_folds = [x for i,x in enumerate(folds) if i != index]
    return [inner for outer in using_folds for inner in outer]


def get_neg_proportion(training_data):
    count =0
    for _,tag in training_data:
        if tag == config.NEGATIVE_TAG:
            count += 1
    return count / len(training_data)
