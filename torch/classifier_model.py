import torch.nn as nn

class LSTMSentimentClassifier(nn.Module):
    def __init__(self, hidden_dim, input_shape,  num_classes):
        super(SentimentClassifier, self).__init__()
        self.rnn = nn.LSTM(input_shape, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        output, _ = self.rnn(x)
        last_hidden = output[:, -1, :]
        logits = self.fc(last_hidden)
        return logits
    

class SentimentClassifier(nn.Module):
    def __init__(self, layers, hidden_dim, dropout_rate, input_shape, num_classes):
        super(SentimentClassifier, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.append(nn.Dropout(dropout_rate))
        self.sequential.append(nn.Linear(in_features=input_shape, out_features=hidden_dim))
        self.sequential.append(nn.ReLU())
        for _ in range(layers - 2):
            self.sequential.append(nn.Dropout(dropout_rate))
            self.sequential.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            self.sequential.append(nn.ReLU())
        self.sequential.append(nn.Dropout(dropout_rate))
        self.sequential.append(nn.Linear(in_features=hidden_dim, out_features=num_classes))

    def forward(self, x):
        x = self.sequential(x)
        return x.view(x.shape[0], -1)