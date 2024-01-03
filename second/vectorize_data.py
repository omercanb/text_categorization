from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

NGRAM_RANGE = (1, 2)
TOP_K = 20000
MIN_DOCUMENT_FREQUENCY = 2
TOKEN_MODE = 'word'

def ngram_vectorize(train_texts, train_labels, val_texts):

    kwargs = {
        'decode_error' : 'replace',
        'strip_accents' : 'unicode',
        'ngram_range' : NGRAM_RANGE,
        'min_df' : MIN_DOCUMENT_FREQUENCY,
        'analyzer' : TOKEN_MODE
    }

    vectorizer = TfidfVectorizer(**kwargs)
    
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)

    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))

    x_train = selector.fit_transform(x_train, train_labels)
    x_val = selector.transform(x_val)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    return x_train, x_val

