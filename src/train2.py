from collections import Counter
#from preprocessing import standardization, data_preprocessing
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import KeyedVectors
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dropout, Dense, Bidirectional,  Flatten, Input, GRU
import matplotlib as mpl
from keras.optimizers import Adam
import pandas as pd
import numpy
from preprocessing import data_preprocessing, external_data
import  tensorflow as tf
from sklearn.model_selection import KFold
from keras.layers import Convolution1D, MaxoutDense, GlobalMaxPooling1D, Input, Conv1D, MaxPooling1D, Flatten, ConvLSTM2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import os

tf.logging.set_verbosity(tf.logging.ERROR)

import gc

#mpl.use('TkAgg')  # or whatever other backend that you want
#import matplotlib.pyplot as plt
np.random.seed(7)
from keras.models import load_model


EMBEDDING_FILE="../GoogleNews-vectors-negative300.bin"
EMBEDDING_DIM=300

#corpora_external="/home/bouche_a/semeval2019/external_data.txt"
corpora_train="/home/bouche_a/semeval2019/data/train.txt"
corpora_test="/home/bouche_a/semeval2019/data/test.txt"


#x_train_extr, y_train_extr = external_data(corpora_external)
x_train, y_train = data_preprocessing(corpora_train, 'True')
x_test= data_preprocessing(corpora_test, 'False')


#x_train=pd.concat([x_train_extr,x_train_sem ])
#y_train=pd.concat([y_train_extr,y_train_sem ])



all_tweet = x_train.append(x_test)



tokenizer = Tokenizer(filters=' ')
tokenizer.fit_on_texts(all_tweet)
word_index = tokenizer.word_index


sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_test = tokenizer.texts_to_sequences(x_test)

sequences = sequences_train + sequences_test

MAX_SEQUENCE_LENGTH = 0
for elt in sequences:
	if len(elt) > MAX_SEQUENCE_LENGTH:
		MAX_SEQUENCE_LENGTH = len(elt)

print(MAX_SEQUENCE_LENGTH)

data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)



indices_train = np.arange(data_train.shape[0])
data_train = data_train[indices_train]

indices_test = np.arange(data_test.shape[0])
data_test = data_test[indices_test]

#indices_train = np.arange(data_train.shape[0])
#data_train = data_train[indices_train]
#labels=sorted(list(set(y_train.tolist())))

#one_hot=np.zeros((len(labels),len(labels)),dtype=int)
#np.fill_diagonal(one_hot,1)
#label_dict=dict(zip(labels,one_hot))
#y_train = y_train.apply(lambda y:label_dict[y]).tolist()


nb_words=len(word_index)+1

y_train = to_categorical(np.asarray(y_train), 4)


embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

oov=[]
oov.append((np.random.rand(EMBEDDING_DIM) * 2.0) - 1.0)
oov = oov / np.linalg.norm(oov)


for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
    else:
        embedding_matrix[i] = oov

split_idx = int(len(x_train)*0.999999999)
x_train, x_val = data_train[:split_idx], data_train[split_idx:]
y_train, y_val = y_train [:split_idx], y_train[split_idx:]

#print('training set: ' + str(len(x_train)) + ' samples')
#print('validation set: ' + str(len(x_val)) + ' samples')

embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True, name='embedding_layer')


def model1(x_train, y_train, x_val, y_val, embedding_layer):

	model1 = Sequential()
	model1.add(embedding_layer)
	model1.add(Dropout(0.5))
	model1.add(GRU(128))
	model1.add(Dropout(0.5))
	model1.add(Dense(32, activation='relu'))
	model1.add(Dropout(0.2))
	model1.add(Dense(4, activation='softmax'))
	model1.compile(loss='categorical_crossentropy',
			      optimizer='Adam',
			      metrics=['acc'])
	model1.summary()
#	early_stopping = EarlyStopping(patience = 2)
#	model_checkpoint = ModelCheckpoint("models", save_best_only = True, save_weights_only = True)
	model1.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=50, epochs=3,  verbose=1)
	model1.save("./model1.h5")

	return model1



def model2(x_train, y_train, x_val, y_val, embedding_layer):
	model2 = Sequential()
	model2.add(embedding_layer)
	#model2.add(Bidirectional(LSTM(256,return_sequences=True)))
	model2.add(Dropout(0.5))
	model2.add(Bidirectional(LSTM(128,return_sequences=True)))
	#model2.add(Dropout(0.5))
#	model2.add(Bidirectional(LSTM(128,return_sequences=True)))
	model2.add(Dropout(0.5))
        #^Bmodel2.add(Dense(256,name='dense1'))
        #model2.add(LeakyReLU(alpha=0.05))
	# model2.add(Dropout(0.5))
	#model2.add(Dense(128,activation='relu', name='dense2'))
	#model2.add(LeakyReLU(alpha=0.05))
	#model2.add(Dropout(0.5))
	model2.add(Dense(64, activation='relu', name='dense3'))
	#model2.add(LeakyReLU(alpha=0.05))
	model2.add(Flatten())
	model2.add(Dense(4,activation='softmax',name='output_6'))
	model2.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
	model2.summary()
	model2.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=50,  verbose=1)
	model2.save("./model2.h5")
	return model2





def CNN1 (x_train, y_train, x_val, y_val, embedding_layer):
	model = Sequential()
	model.add(embedding_layer)
	model.add(Convolution1D(nb_filter=128, filter_length=5, border_mode='valid', activation='relu'))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(4, activation='softmax'))
	model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['acc'])
	model.summary()
	model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=5, batch_size=100)
	model.save("./model3.h5")
	return model




def CNN2 (x_train, y_train, x_val, y_val, embedding_layer ):
	model = Sequential()
	model.add(embedding_layer)
	model.add(Convolution1D(64, 3, border_mode='same'))
	model.add(Convolution1D(32, 3, border_mode='same'))
	model.add(Convolution1D(16, 3, border_mode='same'))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(180,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(4,activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
	model.summary()
	model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=3, batch_size=100)
	model.save("./model4.h5")
	return model






model=model1(x_train, y_train, x_val, y_val,  embedding_layer)
#model=model2(x_train, y_train, x_val, y_val,  embedding_layer)
#model=CNN1(x_train, y_train, x_val, y_val, embedding_layer)
#model=CNN2(x_train, y_train, x_val, y_val, embedding_layer)
"""
cvscores = []
kf=KFold(n_splits=5)
for epoch in range(1,nb_epochs):
	
	print ("======= epoch =", epoch)
	i=1
	cvscores = []
	for train_index, test_index in kf.split(data_train):
		x_train_k, x_val_k = data_train[train_index], data_train[test_index]
		y_train_k, y_val_k = y_train[train_index], y_train[test_index]
	#	print("======================= Fold", i
		model=model2(x_train_k, y_train_k, epoch,  embedding_layer)
		scores = model.evaluate(x_val_k, y_val_k, verbose=0)
		print("Fold", i,  "-->", model.metrics_names[1], scores[1]*100)
#		print("%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		i=i+1
		del model
		gc.collect()
		#keras.clear_session()
		#gc.collect()
	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

"""

f = open("./test.txt", "w")
f.write("id\tturn1\tturn2\tturn3\tlabel\n")
#print ("id\tturn1\tturn2\tturn3\tlabel")
nn_model=load_model("./model1.h5")
r = nn_model.predict(data_test)
print(r)
data = pd.read_csv("/home/bouche_a/semeval2019/data/test.txt", sep='\t', encoding='utf-8',     names=['id','turn1','turn2','turn3'])
 
for d in range(1,len(data)):
	i=d-1 
	idx=numpy.argmax(r[i])
	if (idx==0):
		label="angry"
	elif(idx==1):
		label="happy"
	elif(idx==3):
		label="others"
	elif(idx==2):
		 label="sad"
	f.write(str(data["id"][d])+"\t"+str(data["turn1"][d])+"\t"+str(data["turn2"][d])+"\t"+str(data["turn3"][d])+"\t"+label+"\n")
#	print(str(data["id"][d])+"\t"+str(data["turn1"][d])+"\t"+str(data["turn2"][d])+"\t"+str (data["turn3"][d])+"\t"+label)
f.close()





