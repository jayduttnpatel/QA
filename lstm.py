'''
This file is temp file. Created just to understand the working of the keras in python in LSTM. Here I am doing the sentiment analysis of the
subtitle where the positive or the negative rating was generated randomly in the data file subtitle function.

It is not required in the working of the final project. It is for the understanding purpose only.
'''
import WordToNum
import data
from sklearn.model_selection import train_test_split

mapping=WordToNum.WordToNum()
x,y= data.subtitle(mapping)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

max_words=20000

#import numpy as np

#average_length = np.mean([len(x) for x in x_train])
#median_length = sorted([len(x) for x in x_train])[len(x_train) // 2]

#print("Average sequence length: ", average_length)
#print("Median sequence length: ", median_length)

max_sequence_length = 500

from keras.preprocessing import sequence

x_train = sequence.pad_sequences(x_train, maxlen=max_sequence_length, padding='post', truncating='post')
x_test = sequence.pad_sequences(x_test, maxlen=max_sequence_length, padding='post', truncating='post')

print('X_train shape: ', x_train.shape)

from keras.models import Sequential

from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Bidirectional

# Single layer LSTM example

hidden_size = 32

sl_model = Sequential()
sl_model.add(Embedding(max_words, hidden_size))
sl_model.add(Bidirectional(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2)))
sl_model.add(Dense(1, activation='sigmoid'))
sl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 20

sl_model.fit(x_train, y_train, epochs=epochs, shuffle=True)
loss, acc = sl_model.evaluate(x_test, y_test)

d_model = Sequential()
d_model.add(Embedding(max_words, hidden_size))
d_model.add(Bidirectional(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
d_model.add(Bidirectional(LSTM(hidden_size, activation='tanh', dropout=0.2, recurrent_dropout=0.2)))
d_model.add(Dense(1, activation='sigmoid'))
d_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

d_model.fit(x_train, y_train, epochs=epochs, shuffle=True)
d_loss, d_acc = d_model.evaluate(x_test, y_test)

print('Single layer model -- ACC {} -- LOSS {}'.format(acc, loss))
print('Double layer model -- ACC {} -- LOSS {}'.format(d_acc, d_loss))