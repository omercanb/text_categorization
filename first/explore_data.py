import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def get_num_words_per_sample(texts):
    num_words = [len(text.split()) for text in texts]
    return np.median(num_words)


def plot_sample_length_distribution(sample_texts):
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel("Length of a sample")
    plt.ylabel("Number of samples")
    plt.title("Sample Length Histogram")
    plt.show()



def plot_ngram_frequency(sample_texts):
    max_to_display = 50
    kwargs = {
        "ngram_range" : (1,2),
        "decode_error" : "replace",
        "strip_accents" : "unicode",
        "dtype" : "int32",
        "analyzer" : "word"
    }

    vectorizer = CountVectorizer(**kwargs)
    
    vectorized_texts = vectorizer.fit_transform(sample_texts)

    all_counts = vectorized_texts.sum(axis=0).tolist()[0]
    all_ngrams = list(vectorizer.get_feature_names_out())

    counts, n_grams = zip(*[(c, n) for c, n in sorted(
        zip(all_counts, all_ngrams), reverse=True)][:max_to_display])

    idx = np.arange(max_to_display)
    plt.bar(idx, counts, width=0.8, color='r')
    plt.xlabel("N-gram")
    plt.ylabel("Count")
    plt.title("Freq distribution of n-grams")
    plt.xticks(idx, n_grams, rotation=45)
    plt.show()


def get_num_classes(labels):
    return max(labels) + 1