from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Sample tweets
tweets = ["I love the new product!", "The update is terrible!", "Not bad, but could be better."]
labels = [1, 0, 1]  # 1 = Positive, 0 = Negative

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
seqs = tokenizer.texts_to_sequences(tweets)
padded = pad_sequences(seqs, maxlen=5)

# RNN model
model = Sequential([
    Embedding(input_dim=50, output_dim=8, input_length=5),
    LSTM(10),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded, np.array(labels), epochs=10, verbose=0)

# Predict sentiment
preds = (model.predict(padded) > 0.5).astype("int32")

# Create sentiment network graph
G = nx.Graph()
for i, tweet in enumerate(tweets):
    sentiment = "Positive" if preds[i] == 1 else "Negative"
    G.add_node(tweet, sentiment=sentiment)
for i in range(len(tweets)-1):
    G.add_edge(tweets[i], tweets[i+1])

# Draw graph
colors = ['green' if preds[i] == 1 else 'red' for i in range(len(preds))]
nx.draw(G, with_labels=True, node_color=colors, font_size=8)
plt.show()
