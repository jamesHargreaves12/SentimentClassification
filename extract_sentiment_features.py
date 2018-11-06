import re
import get_folds
from nltk.stem import PorterStemmer
punctuation_regex_start_end = re.compile('^[.,\/#!$%\^&\*;:{}=~()]+|[.,\/#!$%\^&\*;:{}=~()]+$')

def get_unigrams_from_file(file):
    word_splits = file.split()
    unigrams = []
    for word in word_splits:
        punc_iter = punctuation_regex_start_end.finditer(word)
        start_punc = []
        end_punc = []
        for punc in punc_iter:
            if punc.span()[0] == 0:
                start_punc.append(punc.group(0))
            else:
                end_punc.append(punc.group(0))
        unigrams.extend(start_punc)
        unigrams.append(punctuation_regex_start_end.sub('', word))
        unigrams.extend(end_punc)
    stemmer = PorterStemmer()
    unigrams = [stemmer.stem(w) for w in unigrams]

    return unigrams

def get_bigrams_from_file(file):
    unigrams = get_unigrams_from_file(file)
    bigrams = [('^^^',unigrams[0])]
    # we use the token ^^^ to indicate the start of a file (have checked and this is not in any of the files)
    for i in range(len(unigrams)-1):
        bigrams.append((unigrams[i],unigrams[i+1]))
    return bigrams


# tagged_files = get_folds.get_n_folds_consequtive(1)[0]
# first_file = tagged_files[0]
# sec_file = tagged_files[1]
# file_data = first_file[0]
# print(get_unigrams_from_file(file_data))
# sec_file = first_file[0]
# print(get_bigrams_from_file(first_file))
