import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re
#��ȡIMDB����
def rm_tags(text):
    re_tag=re.compile(r'<[^>]+>')
    return re_tag.sub('',text)
import os
def read_files(filetype):
    path='C:/Users/Administrator/Desktop/IMDB���ݼ�/aclImdb/'
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

#����token
token=Tokenizer(num_words=3000)
token.fit_on_texts(train_text)

#��Ӱ������ת���������б�
x_train_seq=token.texts_to_sequences(train_text)
x_test_seq=token.texts_to_sequences(test_text)

#�����������б�ĳ��ȶ�Ϊ100
x_train=sequence.pad_sequences(x_train_seq,maxlen=300)
x_test=sequence.pad_sequences(x_test_seq,maxlen=300)
print(type(x_test))
print(type(x_train))

print(type(y_train))
print(type(y_test))


#����Ƕ��㽫�����б�240ת��Ϊ�����б�1.2,5.1,4.2��
from keras.models import Sequential
from keras.layers import Bidirectional,Flatten
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
model=Sequential()
model.add(Embedding(output_dim=128,input_dim=3000,input_length=300))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True),merge_mode='concat'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()

#ģ��ѵ������
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#ѵ��v erbose��ʾѵ�����̣�validation_split=0.2��ѵ�����ݷ�Ϊѵ����80%����֤��20%
train_history=model.fit(x_train,y_train,batch_size=280,epochs=10,verbose=2,validation_split=0.2)

#ģ������
score=model.evaluate(x_test,y_test,verbose=1)
print(score[1])

from keras.models import load_model
model.save('IMDB_LSTM_model.h5')





