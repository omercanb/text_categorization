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