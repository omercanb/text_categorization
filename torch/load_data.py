import os


def load_imdb_sentiment_data(data_path, seed = 123):
    dir_path = os.path.join(data_path, 'aclImdb')
    train_texts = []
    train_labels = []
    train_path = os.path.join(dir_path, 'train')
    for category in ['pos', 'neg']:
        category_path = os.path.join(train_path, category)
        for fname in os.listdir(category_path):
            if fname.endswith('.txt'):
                with open(os.path.join(category_path, fname)) as f:
                    train_texts.append(f.read())
                    train_labels.append(0 if category == 'neg' else 1)


    test_texts = []
    test_labels = []
    test_path = os.path.join(dir_path, 'test')
    for category in ['pos', 'neg']:
        category_path = os.path.join(test_path, category)
        for fname in os.listdir(category_path):
            if fname.endswith('.txt'):
                with open(os.path.join(category_path, fname)) as f:
                    test_texts.append(f.read())
                    test_labels.append(0 if category == 'neg' else 1)


    return ((train_texts, train_labels),
            (test_texts, test_labels))