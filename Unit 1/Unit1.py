# All Import Statements Defined Here
# Note: Do not add to this list.
# ----------------

import sys
assert sys.version_info[0]==3
assert sys.version_info[1] >= 5

from platform import python_version
assert int(python_version().split(".")[1]) >= 5, "Please upgrade your Python version following the instructions in \
    the README.txt file found in the same directory as this notebook. Your Python version is " + python_version()

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

import nltk
nltk.download('reuters') 
#to specify download location, optionally add the argument: download_dir='/specify/desired/path/'
from nltk.corpus import reuters

import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)
# ----------------


# Co-Occurrence vector : how often some word occurs in a document
def read_corpus(category = "gold"):
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

reuters.corpus = read_corpus()
pprint.pprint(reuters.corpus[:3], compact = True, width = 100)

def distinct_words(corpus):
    """
    Pseudo-code explanation
    For each document:
        for each word in a given document:
            add the word to the set -> set theory -> one of each word (element)
    
    Sort the list of corpus words 
    then, get length
    return corpus word set, and number of corpus words
    """
    corpus_words = []
    n_corpus_words = -1

    corpus_words = set()
    for document in corpus:
        for word in document:
            corpus_words.add(word)
    
    corpus_words = sorted(list(corpus_words))
    n_corpus_words = len(corpus_words)
    return corpus_words, n_corpus_words

def co_occurrence_matrix(corpus, window_size):
    """
    Pseudo-code explanation
    We initialize a zero-matrix of dimension = n_words:
        initialize a dictionary to store indexes of corpus words

    For each doc in our corpus:
        for each target word and it's corresponding index:
            get the index of the target word
            initialize a starting/ending point pointer(s)

            for being the range of our (start, end):
                if the index i != j:
                we set the context word at position j 
                get context index
                at the corresponding pos (target index, context_word) 
                    set it equal to 1 // track our co-occurrences with 1/0 binaries
    """
    words, n_words = distinct_words(corpus)
    M = np.zeros((n_words, n_words), dtype = np.float32)
    word_idx = {word: index for index, word in enumerate(words)}

    for doc in corpus:
        for i, target_word in enumerate(doc):
            target_idx = word_idx[target_word]
            start = max(0, i - window_size)
            end = min(len[doc], i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_word = doc[j]
                    context_index = word_idx[context_word]
                    M[target_idx][context_word] += 1
    return M, word_idx 

def reduce_k_dim(M, k):
    n_iters = 10
    print("Running truncated SVD over %i words..." % (M.shape[0]))
    svd = TruncatedSVD(n_iter = n_iters)
    M_reduced = svd.fit_transform(M)
    print("Done.")
    return M_reduced 

def plot_embeddings(M_reduced, word_idx, words): 
    word_ind = [word_idx[word] for word in words]
    word_embed = M_reduced[word_idx]

    plt.figure(figsize = (10, 8))
    plt.figure(figsize = (10, 8))
    plt.scatter(word_embed[:, 0], word_embed[:, 1], marker = 'o', color = 'r', s = 50)

    for i, word in enumerate(words):
        plt.annotate(word, (word_embed[i, 0], word_embed[i, 1]), fontsize = 12)
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Word Embeddings")


if __name__ == "__main__":
    test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), 
                   "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]

    M_test, word2_ind = co_occurrence_matrix(test_corpus, window_size = 1)
        