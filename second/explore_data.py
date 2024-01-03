import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def get_num_classes(labels):
    return max(labels) + 1


def get_num_samples(sample_texts):
    return len(sample_texts)


def get_number_of_words_per_sample(sample_texts):
    number_of_words = [len(text) for text in sample_texts]
    return np.median(number_of_words)


def plot_class_distribution(labels):
    count_map = Counter(labels)
    num_classes = get_num_classes(labels)
    counts = [count_map[i] for i in range(num_classes)]
    idx = np.arange(num_classes)
    plt.bar(idx, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(idx, idx) # Make the labels 1, 2 instead of 0.25 incremented
    plt.show()


def plot_sample_length_distribution(sample_texts):
    number_of_words = [len(text) for text in sample_texts]
    plt.hist(number_of_words, 50)
    plt.xlabel('Length of sample')
    plt.ylabel('Number of occurances')
    plt.title('Sample Length Distribution')
    plt.show()


def plot_frequency_distribution_of_ngrams(sample_texts,
                                          ngram_range = (1,2),
                                          num_ngrams = 50,):
    kwargs = {
        'ngram_range': ngram_range,
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',  # Split text into word tokens.
    }

    vectorizer = CountVectorizer(**kwargs)

    vectorized_texts = vectorizer.fit_transform(sample_texts)

    all_ngrams = list(vectorizer.get_feature_names_out())
    all_counts = vectorized_texts.sum(axis=0).tolist()[0]

    num_ngrams = min(num_ngrams, len(all_ngrams))

    pairs = sorted(zip(all_counts, all_ngrams), reverse = True)[:num_ngrams]
    counts, ngrams = zip(*pairs)

    idx = np.arange(num_ngrams)
    plt.bar(idx, counts)
    plt.xticks(idx, ngrams, rotation=45)
    plt.xlabel('N-grams')
    plt.ylabel('Frequencies')
    plt.title('Distribution of n-gram frequencies')
    plt.show()