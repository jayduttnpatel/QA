import data
import WordToNum
import data
from sklearn.model_selection import train_test_split

mapping=WordToNum.WordToNum()

sub_train,q_train,t_train,a_train,b_train,c_train,d_train=data.data(mapping,"/temp/train")
sub_test1,q_test1,t_test1,a_test1,b_test1,c_test1,d_test1= data.data(mapping,"/temp/test_NR")
sub_test2,q_test2,t_test2,a_test2,b_test2,c_test2,d_test2= data.data(mapping,"/temp/test_R")

'''import numpy as np

average_length_sub = np.mean([len(x) for x in sub])
median_length_sub = sorted([len(x) for x in sub])[len(sub) // 2]

average_length_q = np.mean([len(x) for x in q])
median_length_q = sorted([len(x) for x in q])[len(q) // 2]

average_length_a = np.mean([len(x) for x in a])
median_length_a = sorted([len(x) for x in a])[len(a) // 2]

average_length_b = np.mean([len(x) for x in b])
median_length_b = sorted([len(x) for x in b])[len(b) // 2]

average_length_c = np.mean([len(x) for x in c])
median_length_c = sorted([len(x) for x in c])[len(c) // 2]

average_length_d = np.mean([len(x) for x in d])
median_length_d = sorted([len(x) for x in d])[len(d) // 2]

print("Average sub length: ", average_length_sub)
print("Median sub length: ", median_length_sub)

print("Average q length: ", average_length_q)
print("Median q length: ", median_length_q)

print("Average a length: ", average_length_a)
print("Median a length: ", median_length_a)

print("Average b length: ", average_length_b)
print("Median b length: ", median_length_b)

print("Average c length: ", average_length_c)
print("Median c length: ", median_length_c)

print("Average d length: ", average_length_d)
print("Median d length: ", median_length_d) '''

sub_max_sequence_length = 500
q_max_sequence_length = 6
a_max_sequence_length = 2
b_max_sequence_length = 2
c_max_sequence_length = 2
d_max_sequence_length = 3
max_words=10000
drop_out=0.5

import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Sequential

from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.models import Model
from sklearn.model_selection import train_test_split

keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              verbose=0, mode='auto')

sub_train = sequence.pad_sequences(sub_train, maxlen=sub_max_sequence_length, padding='post', truncating='post')
q_train = sequence.pad_sequences(q_train, maxlen=q_max_sequence_length, padding='post', truncating='post')
a_train = sequence.pad_sequences(a_train, maxlen=a_max_sequence_length, padding='post', truncating='post')
b_train = sequence.pad_sequences(b_train, maxlen=b_max_sequence_length, padding='post', truncating='post')
c_train = sequence.pad_sequences(c_train, maxlen=c_max_sequence_length, padding='post', truncating='post')
d_train = sequence.pad_sequences(d_train, maxlen=d_max_sequence_length, padding='post', truncating='post')

sub_test1 = sequence.pad_sequences(sub_test1, maxlen=sub_max_sequence_length, padding='post', truncating='post')
q_test1 = sequence.pad_sequences(q_test1, maxlen=q_max_sequence_length, padding='post', truncating='post')
a_test1 = sequence.pad_sequences(a_test1, maxlen=a_max_sequence_length, padding='post', truncating='post')
b_test1 = sequence.pad_sequences(b_test1, maxlen=b_max_sequence_length, padding='post', truncating='post')
c_test1 = sequence.pad_sequences(c_test1, maxlen=c_max_sequence_length, padding='post', truncating='post')
d_test1 = sequence.pad_sequences(d_test1, maxlen=d_max_sequence_length, padding='post', truncating='post')

sub_test2 = sequence.pad_sequences(sub_test2, maxlen=sub_max_sequence_length, padding='post', truncating='post')
q_test2 = sequence.pad_sequences(q_test2, maxlen=q_max_sequence_length, padding='post', truncating='post')
a_test2 = sequence.pad_sequences(a_test2, maxlen=a_max_sequence_length, padding='post', truncating='post')
b_test2 = sequence.pad_sequences(b_test2, maxlen=b_max_sequence_length, padding='post', truncating='post')
c_test2 = sequence.pad_sequences(c_test2, maxlen=c_max_sequence_length, padding='post', truncating='post')
d_test2 = sequence.pad_sequences(d_test2, maxlen=d_max_sequence_length, padding='post', truncating='post')

sub_input = Input(shape=(500,),dtype='int32',name='sub_input')
sub = Embedding(output_dim=512 , input_dim=10000, input_length=500)(sub_input)
#sub_out = LSTM(32)(sub)
sub_out = Bidirectional(LSTM(32))(sub)
sub_out = Dropout(drop_out)(sub_out)

q_input = Input(shape=(6,),dtype='int32',name='q_input')
q = Embedding(output_dim=512 , input_dim=10000, input_length=6)(q_input)
#q_out = LSTM(32)(q)
q_out = Bidirectional(LSTM(32))(q)
q_out = Dropout(drop_out)(q_out)

a_input = Input(shape=(2,),dtype='int32',name='a_input')
a = Embedding(output_dim=512 , input_dim=10000, input_length=2)(a_input)
#a_out = LSTM(32)(a)
a_out = Bidirectional(LSTM(32))(a)
a_out = Dropout(drop_out)(a_out)

b_input = Input(shape=(2,),dtype='int32',name='b_input')
b = Embedding(output_dim=512 , input_dim=10000, input_length=2)(b_input)
#b_out = LSTM(32)(b)
b_out = Bidirectional(LSTM(32))(b)
b_out = Dropout(drop_out)(b_out)

c_input = Input(shape=(2,),dtype='int32',name='c_input')
c = Embedding(output_dim=512 , input_dim=10000, input_length=2)(c_input)
#c_out = LSTM(32)(c)
c_out = Bidirectional(LSTM(32))(c)
c_out = Dropout(drop_out)(c_out)

d_input = Input(shape=(3,),dtype='int32',name='d_input')
d = Embedding(output_dim=512 , input_dim=10000, input_length=3)(d_input)
#d_out = LSTM(32)(d)
d_out = Bidirectional(LSTM(32))(d)
d_out = Dropout(drop_out)(d_out)

x = keras.layers.concatenate([sub_out, q_out, a_out, b_out, c_out, d_out])

x = Dense(64, activation='relu')(x)
x = Dropout(drop_out)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(drop_out)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(drop_out)(x)

main_output = Dense(4, activation='softmax', name='main_output')(x)

model = Model(inputs=[sub_input, q_input, a_input, b_input, c_input, d_input], outputs=[main_output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy')

model.fit([sub_train, q_train, a_train, b_train, c_train, d_train], [t_train],
          epochs=10, batch_size=32, shuffle=True)
		  
test_scores1 = model.evaluate([sub_test1, q_test1, a_test1, b_test1, c_test1, d_test1], [t_test1] )
print('Test loss1:', test_scores1*100)
print('Test acc1:', 100-test_scores1*100)

test_scores2 = model.evaluate([sub_test2, q_test2, a_test2, b_test2, c_test2, d_test2], [t_test2] )
print('Test loss2:', test_scores2*100)
print('Test acc2:', 100-test_scores2*100)
