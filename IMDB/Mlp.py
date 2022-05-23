import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re
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
token=Tokenizer(num_words=3000)
token.fit_on_texts(train_text)

#将影评文字转换成数字列表
x_train_seq=token.texts_to_sequences(train_text)
x_test_seq=token.texts_to_sequences(test_text)

#让所有数字列表的长度都为100
x_train=sequence.pad_sequences(x_train_seq,maxlen=300)
x_test=sequence.pad_sequences(x_test_seq,maxlen=300)
print(type(x_test))
print(type(x_train))

print(type(y_train))
print(type(y_test))


#加入嵌入层将数字列表240转化为向量列表（1.2,5.1,4.2）
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
model=Sequential()
model.add(Embedding(output_dim=128,input_dim=3000,input_length=300))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()

#模型训练函数
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#训练v erbose显示训练过程，validation_split=0.2将训练数据分为训练集80%和验证集20%
train_history=model.fit(x_train,y_train,batch_size=100,epochs=10,verbose=2,validation_split=0.2)

#模型评估
score=model.evaluate(x_test,y_test,verbose=1)
print(score[1])

from keras.models import load_model
model.save('IMDB_mlp_model.h5')





