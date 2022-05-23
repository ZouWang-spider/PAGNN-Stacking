# mnist attention
import numpy as np

np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
import re
from keras.preprocessing import sequence
import keras



#读取IMDB数据
def rm_tags(text):
    re_tag=re.compile(r'<[^>]+>')
    return re_tag.sub('',text)
import os
def read_files(filetype):
    path='C:/Users/Administrator/Desktop/IMDB数据集/aclImdb/'
    file_list=[]
    positive_path=path+filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]

    negative_path=path+filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]

    print('read',filetype,'files:',len(file_list))
    all_labels=([1]*12500+[0]*12500)
    all_texts=[]
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts+=[rm_tags("".join(file_input.readlines()))]
    return all_labels,all_texts

y_train,train_text=read_files("train")
y_test,test_text=read_files("test")
y_train=np.array(y_train)
y_test=np.array(y_test)

#建立token
token=Tokenizer(num_words=3800)
token.fit_on_texts(train_text)

#将影评文字转换成数字列表
x_train_seq=token.texts_to_sequences(train_text)
x_test_seq=token.texts_to_sequences(test_text)

#让所有数字列表的长度都为100
x_train=sequence.pad_sequences(x_train_seq,maxlen=380)
x_test=sequence.pad_sequences(x_test_seq,maxlen=380)
print(type(x_test))
print(type(x_train))

print(type(y_train))
print(type(y_test))



TIME_STEPS = 380
INPUT_DIM = 28
lstm_units = 32


# 第一种 attention
def attention_3d_block(inputs):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)          #置换输入的维度input.shape=(None,28,128)====>a.shape(None,128,28)
    a = Dense(TIME_STEPS, activation='softmax')(a)      #softmax层  (None,128,28)
    a_probs = Permute((2, 1), name='attention_vec')(a)          #(None,28,128)

    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')    #将这两个inputs为LSTM输出的结果*a_probs为softmax输出的结果,对应元素相乘  (None,28,128)
    return output_attention_mul


# build RNN model with attention
inputs = Input(shape=(380,))      #28*28
embed1 = keras.layers.Embedding(3800, 128)(inputs)    #10000  输入的维度为建立字典的单词数量3800,outputdim 128
drop1 = Dropout(0.3)(embed1)
lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.3)(attention_flatten)
output = Dense(1, activation='sigmoid')(drop2)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())



print('Training------------')
hist=model.fit(x_train, y_train, epochs=10, batch_size=280,verbose=2,validation_data=(x_test, y_test))

print('Testing--------------')
loss, accuracy = model.evaluate(x_test, y_test)

print('test loss:', loss)
print('test accuracy:', accuracy)

# 查看训练过程
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

print(hist.history['loss'])
print(hist.history['val_loss'])
print(hist.history['accuracy'])
print(hist.history['val_accuracy'])