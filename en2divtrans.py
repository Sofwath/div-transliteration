import numpy as np

from pathlib import *
import os, sys, io, re, random
from keras.layers import *
from keras.models import *
from keras.optimizers import *

from hyperparams import Hyperparams as hp


def LoadVocab(vocab_fname):
    with io.open(vocab_fname, 'r', encoding='utf-8') as f:
        char_vocab = f.read().splitlines()
    return char_vocab

def ReadXY(training_file, max_word_len=hp.max_seq_len):
    x = []
    y = []
    source_line = LoadVocab(hp.source_file)
    target_line = LoadVocab(hp.target_file)
   
    f = io.open(hp.training_file, 'r', encoding='utf-8')
    for line in f:
        line = line.replace('\n', '').strip()
        source_str = line.split('\t')[0]
        target_str = line.split('\t')[1]
        temp_x = []
        temp_y = []
        for i in range(len(source_str)):
            temp_x.append(source_line.index(source_str[i]))
        for i in range(len(target_str)):
            temp_y.append(OneHot(target_line,target_line.index(target_str[i])))
        r = max_word_len - len(source_str)
        for i in range(r):
            temp_x.append(0)
        r = max_word_len - len(target_str)
        for i in range(r):
            temp_y.append(OneHot(target_line,0))
        x.append(temp_x)
        y.append(temp_y)
    f.close()

    x = np.reshape(x, (len(x), max_word_len))
    print('x-shape', x.shape)
    y = np.reshape(y, (len(y), max_word_len, len(target_line)))
    print('y-shape', y.shape)
    return x, y, source_line, target_line

def TrainModel(model_name):
    x, y, x_chars, y_chars = ReadXY(hp.training_file)
    model = MakeModel(hidden_unit=hp.hidden, max_len=hp.max_seq_len, x_vocab=len(x_chars), y_vocab=len(y_chars), embedding_dim=hp.char_emb_dim)
    weights_file = Path('./'+hp.model_name+'.h5')
    if weights_file.is_file():
        model.load_weights(weights_file)
    model.fit(x=x, y=y, batch_size=hp.batch, epochs=hp.epoch, validation_split=0.2, verbose=1)
    model.save_weights(hp.model_name+'.h5')

def MakeModel(hidden_unit, max_len, x_vocab, y_vocab, embedding_dim):
    input_x_char_index = Input(shape=(max_len,), dtype='int32', name='input_x_char_idx')
    embedding = Embedding(input_dim=x_vocab, input_length=max_len, output_dim=embedding_dim, mask_zero=False)(input_x_char_index)
    rnn = Bidirectional(LSTM(hidden_unit, return_sequences=True), input_shape=(max_len, embedding_dim), merge_mode='ave')(embedding)
    output_y_softmax = TimeDistributed(Dense(y_vocab, activation='softmax'), input_shape=(max_len, rnn[2]), name='output_y_softmax')(rnn)
    model = Model(inputs=input_x_char_index, outputs=output_y_softmax)
    opts = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=opts, loss='categorical_crossentropy', metrics=['acc'])
    print('model summary...')
    model.summary()
    return model

def OneHot(list,idx):
	one_hot_vec = np.zeros((1, len(list)))
	np.put(one_hot_vec, idx, 1)
	return one_hot_vec

if __name__ == '__main__':

    # to train un-comment the following line
    print (hp.model_name)
    TrainModel(hp.model_name)
