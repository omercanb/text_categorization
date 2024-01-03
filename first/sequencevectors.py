from torchnlp.encoders.text import StaticTokenizerEncoder, pad_tensor


TOP_K = 20000
MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(train_texts, val_texts):
    tokenizer = StaticTokenizerEncoder(train_texts, tokenize=lambda s: s.split())

    x_train = [tokenizer.encode(text) for text in train_texts]
    x_val = [tokenizer.encode(text) for text in val_texts]

    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH


    x_train = [pad_tensor(x, max_length) if x.shape[0] < max_length else x[-max_length:] for x in x_train]
    x_val = [pad_tensor(x, max_length) if x.shape[0] < max_length else x[-max_length:] for x in x_val]

    return x_train, x_val


from keras.preprocessing import text
from keras.preprocessing import sequence

# Vectorization parameters
# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 500

def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index