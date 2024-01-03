import torch
import os
import random


# class ImdbSentimentDataset(Dataset):
#     def __init__(self, path, split, transform = None, target_transform = None):
#         self.filepaths_and_labels = []
#         dir_path = os.path.join(path, 'aclImdb')
#         split_path = os.path.join(dir_path, split)
#         for category in ['neg', 'pos']:
#             data_path = os.path.join(split_path, category)
#             for fname in os.listdir(data_path):
#                 label = 0 if category == 'neg' else 1
#                 self.filepaths_and_labels.append((os.path.join(data_path, fname), label))

#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.filepaths_and_labels)
    
#     def __getitem__(self, idx):
#         file_path, label = self.filepaths_and_labels[idx]
#         with open(file_path) as f:
#             text = f.read()

#         if self.transform:
#             text = self.transform(text)

#         if self.target_transform:
#             label = self.target_transform(label)

#         return text, label


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

    # random.seed(seed)
    # random.shuffle(train_texts)
    # random.seed(seed)
    # random.shuffle(train_labels)

    return ((train_texts, train_labels),
            (test_texts, test_labels))