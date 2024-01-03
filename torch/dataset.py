from torch.utils.data import Dataset

class ImdbSentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.vectors = texts
        self.labels = labels

    def __len__(self):
        return self.vectors.shape[0]
    
    def __getitem__(self, idx):
        return (self.vectors[idx], self.labels[idx])