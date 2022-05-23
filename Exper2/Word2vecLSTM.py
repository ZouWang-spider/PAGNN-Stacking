from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import logging
import jieba
import pickle
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

#停用词表
f = open('stopwords.txt', )         # 加载停用词
stopwords = [i.replace("\n", "") for i in f.readlines()]    # 停用词表
#print(stopwords)


def del_stop_words(text):
	word_ls = jieba.lcut(text)
	word_ls = [i for i in word_ls if i not in stopwords]
	return word_ls



# 加载语料
#处理标签
filepath= 'online_shopping_10_cats.csv'

df=pd.read_csv(filepath, error_bad_lines=False)
# 标签及词汇
labels, content = list(df['label'].unique()), list(df['content'].unique())






#wordvec2的语料
with open("online_shopping_10_cats.txt", "r", encoding='UTF-8') as s:
    all_sentences = s.readlines()
    #print(all_sentences)

datas = [del_stop_words(data.replace("\n", "")) for data in all_sentences]   # 处理语料





logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # 将日志输出到控制台

model = Word2Vec(datas,     # 上文处理过的全部语料
                 size=100,  # 词向量维度 默认100维
                 min_count=1,  # 词频阈值 词出现的频率 小于这个频率的词 将不予保存
                 window=5  # 窗口大小 表示当前词与预测词在一个句子中的最大距离是多少
                 )
model.save('Word2vec_v1')  # 保存模型




#创建词语字典，并返回word2vec模型中      词语的索引       词向量
def create_dictionaries(model):

    gensim_dict = Dictionary()    # 创建词语词典
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)

    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号

    w2vec = {word: model[word] for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec


model = Word2Vec.load('Word2vec_v1')                  # 加载模型
index_dict, word_vectors= create_dictionaries(model)  # 索引字典index_dict、词向量字典word_vectors
#print(index_dict)





#使用 pickle 存储序列化数据
pkl_name='dictaddvextor'
output = open(pkl_name + ".pkl", 'wb')
pickle.dump(index_dict, output)  # 索引字典
pickle.dump(word_vectors, output)  # 词向量字典
output.close()


#加载词向量数据 并填充词向量矩阵
f = open("dictaddvextor.pkl", 'rb')
index_dict = pickle.load(f)             # 索引字典，{单词: 索引数字}
word_vectors = pickle.load(f)           # 词向量, {单词: 词向量(100维长的数组)}

n_symbols = len(index_dict) + 1       # 索引数字的个数，因为有的词语索引为0，所以+1   11629个
#print(n_symbols)
embedding_weights = np.zeros((n_symbols, 100))      #创建一个n_symbols * 100的0矩阵

#将每个索引都用词向量表示在   n_symbols*100二维矩阵中
for w, index in index_dict.items():                 #从索引为1的词语开始，用词向量填充矩阵
    embedding_weights[index, :] = word_vectors[w]     #词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）
#print(embedding_weights)




#处理数据内容
#构建语料的字典词库
inverse_index_dict={i+1:word for i,word in enumerate(index_dict)}    #反词典 {i为序列号： Word为数据内容}一共有11629个
#print(inverse_index_dict)

#词汇表大小
vocab_size=len(index_dict.keys())                                 #词汇表大小60000
print(vocab_size)



def text_to_index_array(p_new_dic, p_sen):
    if type(p_sen) == list:
        new_sentences = []
        for sen in p_sen:
            new_sen = []
            for word in sen:
                try:
                    new_sen.append(p_new_dic[word])  # 单词转索引数字
                except:
                    new_sen.append(0)  # 索引字典里没有的词转为数字0
            new_sentences.append(new_sen)
        return np.array(new_sentences)   # 转numpy数组
    else:
        new_sentences = []
        sentences = []
        p_sen = p_sen.split(" ")
        for word in p_sen:
            try:
                sentences.append(p_new_dic[word])  # 单词转索引数字
            except:
                sentences.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(sentences)
        return new_sentences


#将内容进行序列填充
x=text_to_index_array(index_dict, all_sentences)
x=pad_sequences(maxlen=100,sequences=x,padding='post',value=0)            #填充大小为100,篇章分类文本较长设置为7000
print(x.shape)        #(62774,100)


#构建标签字典库
label_dictionary={label:i for i,label in enumerate(labels)}              #{'正面': 0, '负面': 1, nan: 2}
with open('label_dict.pk', 'wb') as f:                                   #将标签字典库写入文件中
    pickle.dump(label_dictionary, f)

output_dictionary = {i: labels for i, labels in enumerate(labels)}       #反标签字典{0: '正面', 1: '负面', 2: nan}

#标签类别数量
label_size=len(label_dictionary.keys())                                #标签大小3 正面 反面 空


#将标签进行独热编码处理
y=[[label_dictionary[sent]] for sent in df['label']]
y=[np_utils.to_categorical(label,num_classes=label_size) for label in y]
y=np.array([list(_[0]) for _ in y])
print(y.shape)           #(62774,10)
#print(y)




'''''''''
#利用生成的词向量构建embedding层的初始权重，能够有效减少模型寻优的时间，一定程度上提高准确率。
#加入嵌入层将数字列表240转化为向量列表（1.2,5.1,4.2）
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,GRU
model=Sequential()
model.add(Embedding(input_dim=n_symbols,output_dim=100, mask_zero=True,weights=[embedding_weights],input_length=100))
model.add(Dropout(0.2))
model.add(LSTM(32))   #32
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(label_size,activation='softmax'))
model.summary()
'''''


#数据维度(62774,100)      (62774,10)
from keras.layers import *
from keras.models import *
inputs = Input(shape=(100,))      #28*28
embed=Embedding(input_dim=n_symbols,output_dim=100, mask_zero=True,weights=[embedding_weights],input_length=100)(inputs)
#drop1 = Dropout(0.3)(embed)
lstm_out = Bidirectional(LSTM(64, return_sequences=True), name='bilstm')(embed)
# gru_out=Bidirectional(GRU(64,return_sequences=True),name='bigru')(embed)



attention_flatten = Flatten()(lstm_out)
drop2 = Dropout(0.3)(attention_flatten)
output = Dense(10, activation='softmax')(drop2)
model = Model(inputs=inputs, outputs=output)



#模型训练函数
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# 划分训练集和测试集，此时都是list列表
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.1,random_state=42)


#训练v erbose显示训练过程，validation_split=0.2将训练数据分为训练集80%和验证集20%
hist=model.fit(train_x,train_y,batch_size=32,epochs=20,verbose=1,validation_split=0.2)

#模型评估
score=model.evaluate(test_x,test_y,verbose=1)
print(score[1])

from keras.models import load_model
model.save('Word2vec_LSTM_model.h5')


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
