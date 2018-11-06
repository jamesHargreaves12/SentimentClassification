from get_folds import get_n_folds
import naive_bayes
import config
from aux_funcs import  merge_all_folds_except, get_neg_proportion

folds = get_n_folds(10)
total_correct_count = 0
for i, fold in enumerate(folds):
    print("Fold: " + str(i))
    correct_count = 0
    test = fold
    training = merge_all_folds_except(i, folds)
    neg_proportion = get_neg_proportion(training)
    pos_features, neg_features = naive_bayes.get_feature_to_count(training, config.MINIMUM_APPEARANCE_OF_FEATURE_IN_TRAINING)
    for testcase,tag in test:
        predict_pos = naive_bayes.argmax_is_pos_sentiment(pos_features, neg_features, testcase, config.NB_SMOOTHED, neg_proportion)
        if predict_pos:
            if tag == config.POSITVIE_TAG:
                correct_count += 1
        elif tag == config.NEGATIVE_TAG:
            correct_count += 1
    total_correct_count += correct_count
    print(correct_count)
print("Total Correct = " + str(total_correct_count))
print("Total = " + str(len(merge_all_folds_except(-1, folds))))
# Split method = consecutive
#   Smoothed Naive Bayes on Unigrams = 78.9% (1578/2000) with minimum feature count to be accepted set to 1
#   Smoothed Naive Bayes on Unigrams = 77.8% (1556/2000) with minimum feature count to be accepted set to 4 (as paper)
#   unsmoothed Naive Bayes = 50.15% (1003/2000) with minimum feature count to be accepted set to 4 (as paper)
#   Smoothed Naive Bayes on Bigrams = 82.2% (1644/2000) with minimum feature count to be accepted set to 4
#   Smoothed Naive Bayes on Bigrams = 80.3% (1606/2000) with minimum feature count to be accepted set to 7 (as paper)
# Split method = round robin:
#   Smoothed Naive Bayes on Unigrams = 78.95% (1579/2000) with minimum feature count to be accepted set to 4 (as paper)
#   Unsmoothed Naive Bayes = 50% (1000/2000) with minimum feature count to be accepted set to 4 (as paper)
#   Smoothed Naive Bayes on Bigrams = 80.65% (1613/2000) with minimum feature count to be accepted set to 7 (as paper)
