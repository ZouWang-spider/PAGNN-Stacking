#采取Attention+BiLstm对汽车评论进行情感分析
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Dense,Embedding,Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

np.random.seed(1337)
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras import backend as K


#导入数据
#特征为evaluation 类别为label
def load_data(filepath,input_shape=300):   #20改
    df=pd.read_csv(filepath)

    #标签及词汇
    labels,vocabulary=list(df['label'].unique()),list(df['content'].unique())
    print(labels)
    print(len(vocabulary))     #24904


    #构造字符级别的特征
    string=''
    for word in vocabulary:
        string+=word


    vocabulary=set(string)

    #字典列表
    word_dictionary={word:i+1 for i,word in enumerate(vocabulary)}           #词典  i为序列号 Word为数据内容
    with open('word_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary,f)
    inverse_word_dictionary={i+1:word for i,word in enumerate(vocabulary)}   #反词典 i为序列号 Word为数据内容

    label_dictionary={label:i for i,label in enumerate(labels)}
    print(list(label_dictionary))
    with open('label_dict.pk','wb') as f:
        pickle.dump(label_dictionary,f)
    output_dictionary={i:labels for i,labels in enumerate(labels)}


    #词汇表大小
    vocab_size=len(word_dictionary.keys())
    #print(vocab_size)
    #标签类别数量 2
    label_size=len(label_dictionary.keys())
    #print(label_size)


    #序列填充，按input_shape填充长度不足的按0补充
    x=[[word_dictionary[word] for word in sent] for sent in df['content']]
    x=pad_sequences(maxlen=input_shape,sequences=x,padding='post',value=0)
    y=[[label_dictionary[sent]] for sent in df['label']]
    y=[np_utils.to_categorical(label,num_classes=label_size) for label in y]
    y=np.array([list(_[0]) for _ in y])

    return x,y,output_dictionary,vocab_size,label_size,inverse_word_dictionary


filepath='imdb.csv'                     #文件路径
input_shape=300                   #评论的平均长度
x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath,input_shape)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=42)
print(train_x.shape)   #(20000,300)    text_x (5000,300)
print(train_y.shape)    #(20000,2)     text_y  (5000,2)





#模型输入参数，需要自己根据需要调整
n_units=32
batch_size=32
epochs=10
output_dim=32     #20改32
TIME_STEPS = 100
#创建Attention+biLSTM模型

import tensorflow as tf
from tensorflow import keras

# 第一种 attention
def attention_3d_block(inputs):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)          #置换输入的维度input.shape=(None,28,128)====>a.shape(None,128,28)
    a = Dense(TIME_STEPS, activation='softmax')(a)      #softmax层  (None,128,28)
    a_probs = Permute((2, 1), name='attention_vec')(a)          #(None,28,128)

    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')    #将这两个inputs为LSTM输出的结果*a_probs为softmax输出的结果,对应元素相乘  (None,28,128)
    return output_attention_mul

#第一种Attention机制：多层感知机得到权重
def attention_block_1(inputs, feature_cnt, dim):
    h_block = int(feature_cnt * dim / 32 / 2)
    hidden = Flatten()(inputs)
    while (h_block >= 1):
        h_dim = h_block * 32
        hidden = Dense(h_dim, activation='selu', use_bias=True)(hidden)
        h_block = int(h_block / 2)
    attention = Dense(feature_cnt, activation='softmax', name='attention')(hidden)
    #    attention = Lambda(lambda x:x*category)(attention)
    attention = RepeatVector(dim)(attention)
    attention = Permute((2, 1))(attention)

    attention_out = Multiply()([attention, inputs])
    return attention_out



#第二种Attention机制：提取特征嵌入向量的每一维得到注意力权重
def attention_block_2(inputs,feature_cnt,dim):
    a = Permute((2, 1))(inputs)
    a = Reshape((dim, feature_cnt))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(feature_cnt, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='attention')(a)
    a = RepeatVector(dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_out = Multiply()([inputs, a_probs])
    return attention_out


#第三种Attention机制：给每一维分配注意力权重，最后按特征进行聚合回来
def attention_block_3(inputs,feature_cnt,dim):
    a = Flatten()(inputs)
    a = Dense(feature_cnt*dim,activation='softmax')(a)
    a = Reshape((feature_cnt,dim,))(a)
    a = Lambda(lambda x: K.sum(x, axis=2), name='attention')(a)
    a = RepeatVector(dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    attention_out = Multiply()([inputs, a_probs])
    return attention_out





#构建模型
inputs = keras.layers.Input(shape=(300,))
embed1 = keras.layers.Embedding(vocab_size+1, 32)(inputs)    #input_dim=vocab_size+1, output_dim=32
print(embed1.shape)
bilstm=keras.layers.Bidirectional(LSTM(n_units, return_sequences=True), name='bilstm')(embed1)
#第一种Attention机制：多层感知机得到权重
#hidden = attention_block_1(bilstm,100,64)
#第二种Attention机制：提取特征嵌入向量的每一维得到注意力权重
#hidden = attention_block_2(bilstm,100,64)
#第三种Attention机制：给每一维分配注意力权重，最后按特征进行聚合回来
#hidden = attention_block_3(bilstm,100,64)
#attention_mul=attention_3d_block(bilstm)
#hidden = Lambda(lambda x:K.mean(x,axis = 1))(hidden)          #平均
f1=keras.layers.Flatten()(bilstm)                              #拼接
drop1=keras.layers.Dropout(0.2)(f1)
output = keras.layers.Dense(2, activation='softmax')(drop1)
model = keras.models.Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



print('Training------------')
hist = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2,validation_data=(test_x, test_y))  # 训练数据上按batch进行一定次数的迭代训练，以拟合网络


print('Testing--------------')
loss, accuracy = model.evaluate(test_x, test_y)


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





