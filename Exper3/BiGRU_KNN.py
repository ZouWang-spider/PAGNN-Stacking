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

#数据的读取与处理，采用Word2Vec对数据预处理，将文本转化为词向量，标签转化为One-hot编码
#停用词表
f = open('E:/Pythonproject/LSTM&Stacking/stopwords.txt', )         # 加载停用词
stopwords = [i.replace("\n", "") for i in f.readlines()]    # 停用词表
#print(stopwords)


def del_stop_words(text):
	word_ls = jieba.lcut(text)
	word_ls = [i for i in word_ls if i not in stopwords]
	return word_ls



# 加载语料
#处理标签
filepath= 'E:/Pythonproject/LSTM&Stacking/online2.csv'

df=pd.read_csv(filepath, error_bad_lines=False)
# 标签及词汇
labels, content = list(df['id'].unique()), list(df['content'].unique())






#wordvec2的语料
with open("E:/Pythonproject/LSTM&Stacking/online_shopping_10_cats.txt", "r", encoding='UTF-8') as s:
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
print(x.shape)        #(62000,100)


#构建标签字典库
label_dictionary={label:i for i,label in enumerate(labels)}              #{'正面': 0, '负面': 1, nan: 2}
with open('label_dict.pk', 'wb') as f:                                   #将标签字典库写入文件中
    pickle.dump(label_dictionary, f)

output_dictionary = {i: labels for i, labels in enumerate(labels)}       #反标签字典{0: '正面', 1: '负面', 2: nan}

#标签类别数量
label_size=len(label_dictionary.keys())                                #标签大小3 正面 反面 空



#将标签进行独热编码处理
y=[[label_dictionary[sent]] for sent in df['id']]
y=[np_utils.to_categorical(label,num_classes=label_size) for label in y]
y=np.array([list(_[0]) for _ in y])
print(y.shape)          #(62000,10)
#print(y)


# 划分训练集和测试集（49600,100）(12400,100)  （49600,）(12400,)
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=42)


import numpy as np
from sklearn.model_selection import KFold

#BiGRU模型
from keras.layers import *
from keras.models import *
import tensorflow as tf
from tensorflow import keras
#BiGRU模型
inputs = Input(shape=(100,))
embed=Embedding(input_dim=n_symbols,output_dim=100, mask_zero=True,weights=[embedding_weights],input_length=100)(inputs)
#drop1 = Dropout(0.3)(embed)
#lstm_out = Bidirectional(LSTM(64, return_sequences=True), name='bilstm')(embed)
gru_out=Bidirectional(GRU(64,return_sequences=True),name='bigru')(embed)
#out = Dropout(0.2)(lstm_out)
attention_flatten = Flatten()(gru_out)
drop2 = Dropout(0.3)(attention_flatten)
output = Dense(10, activation='softmax')(drop2)
model2 = Model(inputs=inputs, outputs=output)

#将One-Hot编码反转为id
def reverse_onehot(onehot_data):
    # onehot_data assumed to be channel last
    data_copy = np.zeros(onehot_data.shape[:-1])
    for c in range(onehot_data.shape[-1]):
        img_c = onehot_data[..., c]
        data_copy[img_c == 1] = c
    return data_copy




#采用KFold对数据交叉处理,并使用BiLSTM和BiGRU模型对数据集进行预测
from sklearn.model_selection import KFold
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):     #n_folds=10
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]     #49600   12400
    second_level_train_set = np.zeros((train_num,))             #产生1行120列的0矩阵，用于存储预测数据12个为一组
    second_level_test_set = np.zeros((test_num,))               #产生1行30列的0矩阵
    test_nfolds_sets = np.zeros((test_num, n_folds))            #产生30行10列的0矩阵
    kf = KFold(n_splits=n_folds)                          #将训练集分为10分,进行交叉训练

    #i从0-9  train_index有120个  test_index有12个
    # 交叉验证需要将train_x训练集划分10份（每份包含12个数据） 将第一份数据作为test其他部分作为train去训练，然后第二份作为test其他为train训练 依次.........................
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]    #(44640,10)   (44640,)
        x_tst, y_tst =  x_train[test_index], y_train[test_index]     #(4960,10)    (4960,)

        #clf.fit(x_tra, y_tra)     #采用5种算法训练
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        clf.fit(x_tra, y_tra, batch_size=32, epochs=10, verbose=1, validation_split=0.2)

        #将预测的One-Hot编码转化为number存储
        train_pred= clf.predict(x_tst)
        number1 = reverse_onehot(train_pred)
        second_level_train_set[test_index] =number1

        test_pred=clf.predict(x_test)
        number2 = reverse_onehot(test_pred)
        test_nfolds_sets[:, i]=number2



        #second_level_train_set[test_index] = clf.predict(x_tst)    #用于存储交叉验证的数据集大小为 49600一行
        #test_nfolds_sets[:,i] = clf.predict(x_test)                #对30样本进行10次预测分别存在每列中，因为采用不同训练集训练10次得到的模型，在预测10次

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)        #用于存储对test_x预测的数据集 大小为1240一行（取平均值后的结果）
    return second_level_train_set, second_level_test_set



train_sets = []
test_sets = []
for clf in [model2]:
    print('BiGRU模型开始运行')
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)
    print('模型运行结束')

meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)   #交叉验证的预测结果
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)     #对test_x的预测结果
#print(meta_train)
#print(meta_test)

#新的数据集meta_train维度(49600,2)  train_y维度(49600,10)需要转转化   meta_test(12400,2)   test_y(12400,10)需要转维度
#我们这里使用5个分类算法分别对新组合的数据集进行训练、预测
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


#对train_y维度(49600,10)需要转转化为（49600，）
train_y2=reverse_onehot(train_y)
#test_y(12400,10)需要转维度转化为(12400,)
test_y2=reverse_onehot(test_y)

knn_model=KNeighborsClassifier()                     #KNN
knn_model.fit(meta_train,train_y2)
KNN_pred=knn_model.predict(meta_test)


#交叉验证
from sklearn.model_selection import cross_val_score
print('交叉验证结果:')
print(cross_val_score(knn_model, meta_test, test_y2,cv=3))

#衡量标准
from sklearn.metrics import classification_report
print('精度、召回率、F1值：')
print(classification_report(test_y2,KNN_pred))







