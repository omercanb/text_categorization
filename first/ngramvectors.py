from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

NGRAM_RANGE = (1, 2)
TOP_K = 20000
MIN_DOCUMENT_FREQUENCY = 2
TOKEN_MODE = 'word'


def ngram_vectorize(train_texts, train_labels, val_texts):
    kwargs = {
        'ngram_range' : NGRAM_RANGE,
        'dtype' : 'int32',
        'decode_error' : 'replace',
        'strip_accents' : 'unicode',
        'analyzer' : TOKEN_MODE,
        'min_df' : MIN_DOCUMENT_FREQUENCY,

    }

    vectorizer = TfidfVectorizer(**kwargs)
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)

    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')

    return x_train, x_val
