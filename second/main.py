import load_data
import train_ngram_model


data = load_data.load_imdb_sentiment_data('')
train_ngram_model.train_ngram_model(data)