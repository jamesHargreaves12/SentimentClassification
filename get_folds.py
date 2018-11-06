import os
import math
import config
from extract_sentiment_features import get_unigrams_from_file, get_bigrams_from_file

def _get_files(path_to_folder):
    files = []
    for file in os.listdir(path_to_folder):
        if file.endswith('.txt'):
            files.append(open(path_to_folder + file).read())
    return files

def get_features_from_file(file):
    if config.GRAM_TYPE == "Unigram":
        return get_unigrams_from_file(file)
    elif config.GRAM_TYPE == "Bigram":
        return get_bigrams_from_file(file)
    else:
        raise ValueError("config.GRAM_TYPE = " + config.GRAM_TYPE)

def get_n_folds(n):
    tagged_feature_pos = [(get_features_from_file(file),config.POSITVIE_TAG) for file in _get_files(config.POSITIVE_DIR)]
    tagged_feature_neg = [(get_features_from_file(file),config.NEGATIVE_TAG) for file in _get_files(config.NEGATIVE_DIR)]
    if config.FOLD_GENERATION == "Consecutive":
        pos_chunked = _chunks_consec(tagged_feature_pos, n)
        neg_chunked = _chunks_consec(tagged_feature_neg, n)
    elif config.FOLD_GENERATION == "Round_robin":
        pos_chunked = _chunks_rr(tagged_feature_pos, n)
        neg_chunked = _chunks_rr(tagged_feature_neg, n)

#     TODO think about what to do if not a multiple of n
    chunks = []
    for i in range(n):
        chunks.append(pos_chunked[i])
        chunks[-1].extend(neg_chunked[i])
    return chunks

def _chunks_rr(long_list, num_chunks):
    chunks = [[] for x in range(num_chunks)]
    current_chunk = 0
    for item in long_list:
        chunks[current_chunk].append(item)
        current_chunk = (current_chunk+1)%num_chunks
    return chunks


def _chunks_consec(long_list, num_chunks):
    chunk_size = int(math.floor(len(long_list)/num_chunks))
    chunks = []
    for i in range(0, len(long_list), chunk_size):
        chunks.append(long_list[i:i + chunk_size])
    return chunks
