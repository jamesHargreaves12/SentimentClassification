# import svmlight
# import config
# from aux_funcs import  merge_all_folds_except
# from get_folds import get_n_folds
#
# def data_prepare_for_svm(data, feature_mapping):
#     # print(data[0])
#     prepared_data = []
#     for file, tag in data:
#         file_svm_features = []
#         for feature in file:
#             if feature in feature_mapping:
#                 if((feature_mapping[feature],1.0) not in file_svm_features):
#                     file_svm_features.append((feature_mapping[feature],1.0)
#         prepared_data.append((1.0 if tag == config.POSITVIE_TAG else -1.0, file_svm_features))
#     return prepared_data
#
# def get_feature_to_float_mapping(training_data):
#     current_key = 0.0
#     mapping = {}
#     for file,_ in training_data:
#         for feature in file:
#             if not feature in mapping:
#                 mapping[feature] = (current_key, 1)
#                 current_key += 1
#             else:
#                 mapping[feature] = (mapping[feature][0], mapping[feature][1]+1)
#     filtered_mapping = {}
#     for k,v in mapping.items():
#         if v[1] >= config.MINIMUM_APPEARANCE_OF_FEATURE_IN_TRAINING:
#             filtered_mapping[k] = v[0]
#     return filtered_mapping
#
# folds = get_n_folds(10)
# for i, fold in enumerate(folds):
#     print("Fold: " + str(i))
#     training = merge_all_folds_except(i, folds)
#     feature_float_mapping = get_feature_to_float_mapping(training)
#     svm_train = data_prepare_for_svm(training, feature_float_mapping)
#     print(svm_train[0])
#     svm_model = svmlight.learn(svm_train, type='classification')
#
# #     for testcase,tag in test:
# #         predict_pos = naive_bayes.argmax_is_pos_sentiment(pos_features, neg_features, testcase, config.NB_SMOOTHED, neg_proportion)
# #         if predict_pos:
# #             if tag == config.POSITVIE_TAG:
# #                 correct_count += 1
# #         elif tag == config.NEGATIVE_TAG:
# #             correct_count += 1
# #     total_correct_count += correct_count
# #     print(correct_count)
# # print("Total Correct = " + str(total_correct_count))
# # print("Total = " + str(len(merge_all_folds_accept(-1, folds))))
