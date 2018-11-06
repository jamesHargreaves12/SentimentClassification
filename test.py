import math
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
unigrams = [stemmer.stem(w) for w in ['this', 'that', 'shopping', '...']]
print(unigrams)