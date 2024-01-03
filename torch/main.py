import load_data
import dataset
import vectorize_data
import classifier_model

import torch.nn as nn
import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

device = 'mps'

(train_texts, train_labels), (test_texts, test_labels) = load_data.load_imdb_sentiment_data('')

x_train, x_test = vectorize_data.ngram_vectorize(train_texts, train_labels, test_texts)
x_train = x_train.todense()
x_test = x_test.todense()


vocab_size = x_train.shape[1]
hidden_dim = 128
num_classes = 2
batch_size = 32
num_epochs = 10
learning_rate = 1e-4

CHECKPOINT_PATH = 'torch/models/classifier'


model = classifier_model.SentimentClassifier(layers=2, hidden_dim=32, dropout_rate=0.3, 
                                             input_shape=vocab_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = dataset.ImdbSentimentDataset(x_train, train_labels)
test_dataset = dataset.ImdbSentimentDataset(x_test, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


for epoch in range(num_epochs):
    total_train_loss = 0
    num_train_samples = 0
    train_total_correct = 0

    model.train()
    pbar = tqdm(total = len(train_loader))
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        guesses = torch.argmax(outputs, dim=1)
        num_correct_outputs = sum(guesses == y)

        train_total_correct += num_correct_outputs
        num_train_samples += x.shape[0]
        total_train_loss += loss.item() * x.shape[0]
        train_accuracy = train_total_correct / num_train_samples

        pbar.set_description(f"Running Accuracy : {train_accuracy:.4f}")
        pbar.update(1)

    checkpoint = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, f'epoch-{epoch}'))


    total_test_loss = 0
    num_test_samples = 0
    test_total_correct = 0

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            val_loss = criterion(outputs, y)

            guesses = torch.argmax(outputs, dim=1)
            num_correct_outputs = sum(guesses == y)

            test_total_correct += num_correct_outputs
            num_test_samples += x.shape[0]
            total_test_loss += val_loss.item() * x.shape[0]
            test_accuracy = test_total_correct / num_test_samples

    train_avg_loss = total_train_loss / num_train_samples
    test_average_loss = total_test_loss / num_test_samples

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_avg_loss:.4f}, Test Loss: {test_average_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    