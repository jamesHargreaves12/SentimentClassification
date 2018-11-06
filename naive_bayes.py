from collections import defaultdict
import config
import math

def process_file_to_count_dict(count_dict, file):
    for feature in file:
        count_dict[feature] = count_dict[feature]+1
    return count_dict

def get_feature_to_count(tagged_training_files, minimum_apperances):
    pos_feature_count = defaultdict(lambda: 0)
    neg_feature_count = defaultdict(lambda: 0)
    for file, tag in tagged_training_files:
        if tag == config.NEGATIVE_TAG:
            process_file_to_count_dict(neg_feature_count, file)
        else:
            process_file_to_count_dict(pos_feature_count, file)
    pos_count_filtered = {k:v for k,v in pos_feature_count.items() if v>=minimum_apperances}
    neg_count_filtered= {k:v for k,v in neg_feature_count.items() if v>=minimum_apperances}
    # TODO this is wrong since a feature is only filtered if it appears less then minimum_apperances in positive + negative
    return pos_count_filtered, neg_count_filtered

def get_total_count(feature_count):
    total_count = 0
    for v in feature_count.values():
        total_count += v
    return total_count


def find_log_prob(file, feature_count, p_sentiment, smoothed, smoothing_constant):
    log_prob = math.log(p_sentiment)
    log_k = math.log(smoothing_constant)
    total_count = get_total_count(feature_count)
    num_features = len(feature_count)
    if smoothed:
        log_denominator = math.log(total_count + num_features * smoothing_constant)
    else:
        log_denominator = math.log(total_count)
    for feature in file:
        if smoothed:
            if feature in feature_count:
                log_numerator = math.log(feature_count[feature]+smoothing_constant)
            else:
                log_numerator = log_k
            log_prob += log_numerator - log_denominator
            # feature_prob = (math.log(feature_count[feature]+smoothing_constant) if feature in feature_count else log_k) - math.log(total_count+len(feature_count)*smoothing_constant)
            # log_prob += feature_prob
            # print(feature + " = " + str(feature_prob))
        else:
            if feature in feature_count:
                log_numerator = math.log(feature_count[feature])
            else:
                log_numerator = float('-inf')
            log_prob += log_numerator - log_denominator
    return log_prob


def argmax_is_pos_sentiment(pos_feature_count, neg_feature_count, file, smoothed, neg_proportion, smoothing_constant=0.01):
    # print("F= "+str(file))
    p_pos = find_log_prob(file, pos_feature_count, 1-neg_proportion, smoothed, smoothing_constant)
    p_neg = find_log_prob(file, neg_feature_count, neg_proportion, smoothed, smoothing_constant)
    # print(p_pos,p_neg)
    return p_pos > p_neg
