import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re
from keras import backend as K
from keras.datasets import  imdb
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


#构建TextCNN模型
#输入句子的长度为256
import tensorflow as tf
from tensorflow import keras
convs = []
inputs = keras.layers.Input(shape=(380,))
embed1 = keras.layers.Embedding(3800, 32)(inputs)    #10000  输入的维度为建立字典的单词数量3800
# embed = keras.layers.Reshape(-1,256, 32, 1)(embed1)
print(embed1[0])

def reshapes(embed1):
    embed = tf.reshape(embed1, [-1, 380, 32, 1])
    return embed

# embed = tf.reshape(embed1, [-1, 256, 32, 1])
embed = keras.layers.Lambda(reshapes)(embed1)
print(embed[0])
l_conv1 = keras.layers.Conv2D(filters=20, kernel_size=(2, 32), activation='relu')(embed)  # 现长度 = 1+（原长度-卷积核大小+2*填充层大小） /步长 卷积核的形状（fsz，embedding_size）
l_pool1 = keras.layers.MaxPooling2D(pool_size=(379, 1))(l_conv1)  # 这里面最大的不同 池化层核的大小与卷积完的数据长度一样255
l_pool11 = keras.layers.Flatten()(l_pool1)  # 一般为卷积网络最近全连接的前一层，用于将数据压缩成一维
convs.append(l_pool11)
l_conv2 = keras.layers.Conv2D(filters=20, kernel_size=(3, 32), activation='relu')(embed)
l_pool2 = keras.layers.MaxPooling2D(pool_size=(378, 1))(l_conv2)     #254
l_pool22 = keras.layers.Flatten()(l_pool2)
convs.append(l_pool22)
l_conv3 = keras.layers.Conv2D(filters=20, kernel_size=(4, 32), activation='relu')(embed)
l_pool3 = keras.layers.MaxPooling2D(pool_size=(377, 1))(l_conv3)    #253
l_pool33 = keras.layers.Flatten()(l_pool3)
convs.append(l_pool33)
merge = keras.layers.concatenate(convs, axis=1)

out = keras.layers.Dropout(0.5)(merge)
shape1=keras.layers.Lambda(lambda out1:K.expand_dims(out1,axis=-1))(out)
print(shape1.shape)

lstm1=keras.layers.LSTM(32)(shape1)                        #添加的LSTM层3维度
out2=keras.layers.Dropout(0.25)(lstm1)                      #添加的Dropout层

output = keras.layers.Dense(32, activation='relu')(out2)
pred = keras.layers.Dense(units=1, activation='sigmoid')(output)

model = keras.models.Model(inputs=inputs, outputs=pred)
model.summary()


#模型训练函数
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#训练v erbose显示训练过程，validation_split=0.2将训练数据分为训练集80%和验证集20%
train_history=model.fit(x_train,y_train,batch_size=280,epochs=10,verbose=2,validation_split=0.2)

#模型评估
score=model.evaluate(x_test,y_test,verbose=1)
print(score[1])

from keras.models import load_model
model.save('IMDB_CNN_LSTM_model.h5')
