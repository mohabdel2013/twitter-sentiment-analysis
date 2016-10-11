import numpy as np
np.random.seed(1)  # for reproducibility

import os, theano 
theano.config.openmp = True

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Embedding, Merge, Dropout 
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D

from keras import backend as K

from sklearn import svm, metrics

TEXT_DATA_DIR 		= 'dataset'
WORD_VECTOR_DIR		= 'vectors'
WORD_VECTOR_FILE	= 'sswe-u.txt'

TRAIN_DIR			= 'train'
TEST_DIR			= 'test'
TRAIN_FEATURE_DIR	= 'train_feature'
TEST_FEATURE_DIR	= 'test_feature'

MAX_SEQUENCE_LENGTH = 40
EMBEDDING_DIM 		= 50
NB_FILTER			= 40
FILTER_SIZE			= 5
TRAINABLE 			= True

BATCH_SIZE 			= 50
NB_EPOCH 			= 5

labels_index 		= { 'negative' : 0, 'positive' : 1}
texts 				= []
labels 				= []

nb_train_samples	= 0
nb_test_samples 	= 0

#Loading train documents
for name in labels_index:
	with open(os.path.join(TEXT_DATA_DIR, TRAIN_DIR, name), 'r') as f:
		for line in f:
			texts.append(line)
			labels.append(labels_index[name])
			nb_train_samples += 1

#Loading test documents
for name in labels_index:
	with open(os.path.join(TEXT_DATA_DIR, TEST_DIR, name), 'r') as f:
		for line in f:
			texts.append(line)
			labels.append(labels_index[name])
			nb_test_samples += 1

feature_train	= np.zeros((nb_train_samples, 40))
feature_test	= np.zeros((nb_test_samples, 40))

#Loading train lexicon features
index = 0
for name in labels_index:
	with open(os.path.join(TEXT_DATA_DIR, TRAIN_FEATURE_DIR, name), 'r') as f:
		for line in f:
			feature_train[index, :] = np.asarray(line.strip().split(' '))
			index += 1

#Loading test lexicon features
index = 0
for name in labels_index:
	with open(os.path.join(TEXT_DATA_DIR, TEST_FEATURE_DIR, name), 'r') as f:
		for line in f:
			feature_test[index, :] = np.asarray(line.strip().split(' '))
			index += 1

#Tokenizing documents
tokenizer = Tokenizer(lower=False, filters='\t\n')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

#Ready data for network
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

x_train = data[:nb_train_samples]
y_train = labels[:nb_train_samples]
x_test = data[nb_train_samples:]
y_test = labels[nb_train_samples:]

#Shuffling train data
indices = np.arange(nb_train_samples)
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]
feature_train = feature_train[indices]

#Loading word vectors
embeddings_index = {}
f = open(os.path.join(WORD_VECTOR_DIR, WORD_VECTOR_FILE))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for i in range(len(word_index)):
	embedding_matrix[i+1] = embeddings_index.get('<unk>')
embedding_matrix[0] = embeddings_index.get('<s>')

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Creating network
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=TRAINABLE))
model.add(Convolution1D(NB_FILTER, FILTER_SIZE, activation='relu'))
model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH - FILTER_SIZE + 1))
model.add(Flatten())
model.add(Dense(NB_FILTER, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(len(labels_index), activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

#Learning network
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE)

y_train = np.argmax(y_train, axis=1)
y_test  = np.argmax(y_test, axis=1)

#this comment is just for test, by Salar
y_pred = np.argmax(model.predict(x_test), axis=1)
print 'Network:'
print '\t Accuracy : %.4f' % metrics.accuracy_score(y_test, y_pred)
print '\t Macro-Average Precision : %.4f' % ((metrics.precision_score(y_test, y_pred, pos_label=0) + metrics.precision_score(y_test, y_pred, pos_label=1))/2)
print '\t Macro-Average Recall : %.4f' % ((metrics.recall_score(y_test, y_pred, pos_label=0) + metrics.recall_score(y_test, y_pred, pos_label=1))/2)
print '\t Macro-Average F1 : %.4f' % ((metrics.f1_score(y_test, y_pred, pos_label=0) + metrics.f1_score(y_test, y_pred, pos_label=1))/2)

#Defining function for output of layer before the last one
extractor = K.function([model.layers[0].input], [model.layers[4].output])

x_train = extractor([x_train])[0]
x_test  = extractor([x_test])[0]

clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print 'Conv Features:'
print '\t Accuracy : %.4f' % metrics.accuracy_score(y_test, y_pred)
print '\t Macro-Average Precision : %.4f' % ((metrics.precision_score(y_test, y_pred, pos_label=0) + metrics.precision_score(y_test, y_pred, pos_label=1))/2)
print '\t Macro-Average Recall : %.4f' % ((metrics.recall_score(y_test, y_pred, pos_label=0) + metrics.recall_score(y_test, y_pred, pos_label=1))/2)
print '\t Macro-Average F1 : %.4f' % ((metrics.f1_score(y_test, y_pred, pos_label=0) + metrics.f1_score(y_test, y_pred, pos_label=1))/2)

clf = svm.SVC()
clf.fit(feature_train, y_train)
y_pred = clf.predict(feature_test)
print 'Lexicon:'
print '\t Accuracy : %.4f' % metrics.accuracy_score(y_test, y_pred)
print '\t Macro-Average Precision : %.4f' % ((metrics.precision_score(y_test, y_pred, pos_label=0) + metrics.precision_score(y_test, y_pred, pos_label=1))/2)
print '\t Macro-Average Recall : %.4f' % ((metrics.recall_score(y_test, y_pred, pos_label=0) + metrics.recall_score(y_test, y_pred, pos_label=1))/2)
print '\t Macro-Average F1 : %.4f' % ((metrics.f1_score(y_test, y_pred, pos_label=0) + metrics.f1_score(y_test, y_pred, pos_label=1))/2)

x_train = np.concatenate((x_train, feature_train), axis=1)
x_test  = np.concatenate((x_test, feature_test), axis=1)

clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print 'All:'
print '\t Accuracy : %.4f' % metrics.accuracy_score(y_test, y_pred)
print '\t Macro-Average Precision : %.4f' % ((metrics.precision_score(y_test, y_pred, pos_label=0) + metrics.precision_score(y_test, y_pred, pos_label=1))/2)
print '\t Macro-Average Recall : %.4f' % ((metrics.recall_score(y_test, y_pred, pos_label=0) + metrics.recall_score(y_test, y_pred, pos_label=1))/2)
print '\t Macro-Average F1 : %.4f' % ((metrics.f1_score(y_test, y_pred, pos_label=0) + metrics.f1_score(y_test, y_pred, pos_label=1))/2)
