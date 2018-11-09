import numpy as np
from pathlib import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *

source_char_file = 'en-chars.txt'
target_char_file = 'div-chars.txt'
training_file = 'transdata.txt'
test_file  = 'testdata1.csv'

hidden = 128
max_seq_len = 35
epoch = 30
batch = 128
char_embedding_dim = 64

model_name = 'en-hi-transliteration'

# trains a model and saves weights in 'model_name.h5, validation data = 20% of total data'
def train(model_name):
    x, y, x_chars, y_chars = read_x_y(training_file)
    model = make_model(hidden_unit=hidden, max_len=max_seq_len, x_vocab=len(x_chars), y_vocab=len(y_chars), embedding_dim=char_embedding_dim)
    weights_file = Path('./'+model_name+'.h5')
    if weights_file.is_file():
        model.load_weights(weights_file)
    model.fit(x=x, y=y, batch_size=batch, epochs=epoch, validation_split=0.2, verbose=1)
    model.save_weights(model_name+'.h5')


# load chars to char list
def load_chars(char_file):
    # assumes one char per line
    char_list = []
    f = open(char_file, 'r', encoding='utf-8')
    for line in f:
        line = line.replace('\n','').strip()
        char_list.append(line)
    f.close()
    return char_list


# makes one hot vector given a list and the index of 1
def make_one_hot(idx, list):
    one_hot_seq = np.zeros((1, len(list)))
    np.put(one_hot_seq, idx, 1)
    return one_hot_seq
    #return to_categorical(list, num_classes=idx)


# creates x,y numpy array from training file
def read_x_y(training_file, max_word_len=max_seq_len):
    x = []
    y = []
    source_chars = load_chars(source_char_file)
    target_chars = load_chars(target_char_file)
    # assumes one word pair tab separated per line
    # source_word<tab>target_word
    f = open(training_file, 'r', encoding='utf-8')
    for line in f:
        line = line.replace('\n', '').strip()
        if len(line) > 0 and len(line.split('\t')) == 2:
            source_str = line.split('\t')[0]
            target_str = line.split('\t')[1]
            temp_x = []
            temp_y = []
            for i in range(len(source_str)):
                temp_x.append(source_chars.index(source_str[i]))
            for i in range(len(target_str)):
                temp_y.append(make_one_hot(target_chars.index(target_str[i]), target_chars))
            # x-padding
            r = max_word_len - len(source_str)
            for i in range(r):
                temp_x.append(0)
            # y-padding
            r = max_word_len - len(target_str)
            for i in range(r):
                temp_y.append(make_one_hot(0, target_chars))
            #temp_x.reverse()
            x.append(temp_x)
            y.append(temp_y)
    f.close()
    x = np.reshape(x, (len(x), max_word_len))
    print('x-shape', x.shape)
    y = np.reshape(y, (len(y), max_word_len, len(target_chars)))
    print('y-shape', y.shape)
    return x, y, source_chars, target_chars

# defines model
def make_model(hidden_unit, max_len, x_vocab, y_vocab, embedding_dim):
    input_x_char_index = Input(shape=(max_len,), dtype='int32', name='input_x_char_idx')
    embedding = Embedding(input_dim=x_vocab, input_length=max_len, output_dim=embedding_dim, mask_zero=False)(input_x_char_index)
    rnn = Bidirectional(LSTM(hidden_unit, return_sequences=True), input_shape=(max_len, embedding_dim), merge_mode='ave')(embedding)
    output_y_softmax = TimeDistributed(Dense(y_vocab, activation='softmax'), input_shape=(max_len, rnn[2]), name='output_y_softmax')(rnn)
    model = Model(inputs=input_x_char_index, outputs=output_y_softmax)
    opts = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=opts, loss='categorical_crossentropy', metrics=['acc'])
    print('model-input ', model.input_shape)
    print('model-params ', model.count_params())
    print('model-output ', model.output_shape)
    print('model-summary ')
    model.summary()
    return model


# predict for a test file
def predict(test_file, max_word_len=max_seq_len):
    x = []
    gold_y = []
    source_chars = load_chars(source_char_file)
    target_chars = load_chars(target_char_file)
    # assumes one word pair tab separated per line
    # source_word<tab>target_word
    f = open(test_file, 'r', encoding='utf-8')
    for line in f:
        line = line.replace(' ', '\n')
        line = line.replace('\n', '').strip()
        temp_x = []
        source_str = line.split('\t')[0]
        for i in range(len(source_str)):
            temp_x.append(source_chars.index(source_str[i]))
        r = max_word_len - len(source_str)
        for i in range(r):
            temp_x.append(0)
        x.append(temp_x)

    f.close()
    x = np.reshape(x, (len(x), max_word_len))
    model = make_model(hidden_unit=hidden, max_len=max_seq_len, x_vocab=len(source_chars), y_vocab=len(target_chars), embedding_dim=char_embedding_dim)
    model.load_weights(model_name+'.h5')
    y = model.predict(x=x, batch_size=batch, verbose=1)
    f = open(test_file+'.out','w', encoding='utf-8')
    for i in range(len(y)):
        yw = y[i]
        xw = x[i]
        strs_x = ''
        for j in range(len(xw)):
            if source_chars[xw[j]] != '<P>':
                strs_x += source_chars[xw[j]]
        strs_x = strs_x.strip()
        strs_y = ''
        for j in range(len(yw)):
            c = target_chars[np.argmax(yw[j], axis=0)]
            if c != '<P>':
                strs_y +=c
        strs_y = strs_y.strip()
        #print (strs_y)
        f.write(strs_x+'\t'+strs_y+'\n')
        f.flush()
    f.close()


# evaluates output given test file and predictions
def evaluate(test_outputs):

    f = open(test_outputs, 'r', encoding='utf-8')
    c = 0
    ic = 0
    for line in f:
        line = line.replace('\n', '').strip()
        if len(line) > 0 and len(line.split('\t')) == 3:
            gold_s = line.split('\t')[1].strip()
            pred_s = line.split('\t')[2].strip()
            if gold_s == pred_s:
                c = c + 1
            else:
                ic = ic + 1
    f.close()
    acc = (100*c)/float(c+ic)
    print('\nAccuracy:', acc)

if __name__ == '__main__':

    # to train un-comment the following line
    #train(model_name)

    # to predict un-comment the following line
    predict(test_file)

    # to evaluate un-comment the following line
    #evaluate(test_file+'.out')