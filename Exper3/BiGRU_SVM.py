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

#���ݵĶ�ȡ�봦������Word2Vec������Ԥ�������ı�ת��Ϊ����������ǩת��ΪOne-hot����
#ͣ�ôʱ�
f = open('E:/Pythonproject/LSTM&Stacking/stopwords.txt', )         # ����ͣ�ô�
stopwords = [i.replace("\n", "") for i in f.readlines()]    # ͣ�ôʱ�
#print(stopwords)


def del_stop_words(text):
	word_ls = jieba.lcut(text)
	word_ls = [i for i in word_ls if i not in stopwords]
	return word_ls



# ��������
#�����ǩ
filepath= 'E:/Pythonproject/LSTM&Stacking/online2.csv'

df=pd.read_csv(filepath, error_bad_lines=False)
# ��ǩ���ʻ�
labels, content = list(df['id'].unique()), list(df['content'].unique())






#wordvec2������
with open("E:/Pythonproject/LSTM&Stacking/online_shopping_10_cats.txt", "r", encoding='UTF-8') as s:
    all_sentences = s.readlines()
    #print(all_sentences)

datas = [del_stop_words(data.replace("\n", "")) for data in all_sentences]   # ��������





logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # ����־���������̨

model = Word2Vec(datas,     # ���Ĵ������ȫ������
                 size=100,  # ������ά�� Ĭ��100ά
                 min_count=1,  # ��Ƶ��ֵ �ʳ��ֵ�Ƶ�� С�����Ƶ�ʵĴ� �����豣��
                 window=5  # ���ڴ�С ��ʾ��ǰ����Ԥ�����һ�������е��������Ƕ���
                 )
model.save('Word2vec_v1')  # ����ģ��




#���������ֵ䣬������word2vecģ����      ���������       ������
def create_dictionaries(model):

    gensim_dict = Dictionary()    # ��������ʵ�
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)

    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # �������������1��ʼ���

    w2vec = {word: model[word] for word in w2indx.keys()}  # ����Ĵ�����
    return w2indx, w2vec


model = Word2Vec.load('Word2vec_v1')                  # ����ģ��
index_dict, word_vectors= create_dictionaries(model)  # �����ֵ�index_dict���������ֵ�word_vectors
#print(index_dict)





#ʹ�� pickle �洢���л�����
pkl_name='dictaddvextor'
output = open(pkl_name + ".pkl", 'wb')
pickle.dump(index_dict, output)  # �����ֵ�
pickle.dump(word_vectors, output)  # �������ֵ�
output.close()


#���ش��������� ��������������
f = open("dictaddvextor.pkl", 'rb')
index_dict = pickle.load(f)             # �����ֵ䣬{����: ��������}
word_vectors = pickle.load(f)           # ������, {����: ������(100ά��������)}

n_symbols = len(index_dict) + 1       # �������ֵĸ�������Ϊ�еĴ�������Ϊ0������+1   11629��
#print(n_symbols)
embedding_weights = np.zeros((n_symbols, 100))      #����һ��n_symbols * 100��0����

#��ÿ���������ô�������ʾ��   n_symbols*100��ά������
for w, index in index_dict.items():                 #������Ϊ1�Ĵ��￪ʼ���ô�����������
    embedding_weights[index, :] = word_vectors[w]     #���������󣬵�һ����0������û������Ϊ0�Ĵ��δ����䣩
#print(embedding_weights)




#������������
#�������ϵ��ֵ�ʿ�
inverse_index_dict={i+1:word for i,word in enumerate(index_dict)}    #���ʵ� {iΪ���кţ� WordΪ��������}һ����11629��
#print(inverse_index_dict)

#�ʻ���С
vocab_size=len(index_dict.keys())                                 #�ʻ���С60000
print(vocab_size)



def text_to_index_array(p_new_dic, p_sen):
    if type(p_sen) == list:
        new_sentences = []
        for sen in p_sen:
            new_sen = []
            for word in sen:
                try:
                    new_sen.append(p_new_dic[word])  # ����ת��������
                except:
                    new_sen.append(0)  # �����ֵ���û�еĴ�תΪ����0
            new_sentences.append(new_sen)
        return np.array(new_sentences)   # תnumpy����
    else:
        new_sentences = []
        sentences = []
        p_sen = p_sen.split(" ")
        for word in p_sen:
            try:
                sentences.append(p_new_dic[word])  # ����ת��������
            except:
                sentences.append(0)  # �����ֵ���û�еĴ�תΪ����0
        new_sentences.append(sentences)
        return new_sentences


#�����ݽ����������
x=text_to_index_array(index_dict, all_sentences)
x=pad_sequences(maxlen=100,sequences=x,padding='post',value=0)            #����СΪ100,ƪ�·����ı��ϳ�����Ϊ7000
print(x.shape)        #(62000,100)


#������ǩ�ֵ��
label_dictionary={label:i for i,label in enumerate(labels)}              #{'����': 0, '����': 1, nan: 2}
with open('label_dict.pk', 'wb') as f:                                   #����ǩ�ֵ��д���ļ���
    pickle.dump(label_dictionary, f)

output_dictionary = {i: labels for i, labels in enumerate(labels)}       #����ǩ�ֵ�{0: '����', 1: '����', 2: nan}

#��ǩ�������
label_size=len(label_dictionary.keys())                                #��ǩ��С3 ���� ���� ��



#����ǩ���ж��ȱ��봦��
y=[[label_dictionary[sent]] for sent in df['id']]
y=[np_utils.to_categorical(label,num_classes=label_size) for label in y]
y=np.array([list(_[0]) for _ in y])
print(y.shape)          #(62000,10)
#print(y)


# ����ѵ�����Ͳ��Լ���49600,100��(12400,100)  ��49600,��(12400,)
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.2,random_state=42)


import numpy as np
from sklearn.model_selection import KFold

#BiGRUģ��
from keras.layers import *
from keras.models import *
import tensorflow as tf
from tensorflow import keras
#BiGRUģ��
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

#��One-Hot���뷴תΪid
def reverse_onehot(onehot_data):
    # onehot_data assumed to be channel last
    data_copy = np.zeros(onehot_data.shape[:-1])
    for c in range(onehot_data.shape[-1]):
        img_c = onehot_data[..., c]
        data_copy[img_c == 1] = c
    return data_copy




#����KFold�����ݽ��洦��,��ʹ��BiLSTM��BiGRUģ�Ͷ����ݼ�����Ԥ��
from sklearn.model_selection import KFold
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):     #n_folds=10
    """
    ���������stacking�ĺ��ģ�ʹ�ý�����֤�ķ����õ��μ�ѵ����
    x_train, y_train, x_test ��ֵӦ��Ϊnumpy������������� numpy.ndarray .
    �������Ϊpandas��DataFrame�������ѱ���"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]     #49600   12400
    second_level_train_set = np.zeros((train_num,))             #����1��120�е�0�������ڴ洢Ԥ������12��Ϊһ��
    second_level_test_set = np.zeros((test_num,))               #����1��30�е�0����
    test_nfolds_sets = np.zeros((test_num, n_folds))            #����30��10�е�0����
    kf = KFold(n_splits=n_folds)                          #��ѵ������Ϊ10��,���н���ѵ��

    #i��0-9  train_index��120��  test_index��12��
    # ������֤��Ҫ��train_xѵ��������10�ݣ�ÿ�ݰ���12�����ݣ� ����һ��������Ϊtest����������Ϊtrainȥѵ����Ȼ��ڶ�����Ϊtest����Ϊtrainѵ�� ����.........................
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]    #(44640,10)   (44640,)
        x_tst, y_tst =  x_train[test_index], y_train[test_index]     #(4960,10)    (4960,)

        #clf.fit(x_tra, y_tra)     #����5���㷨ѵ��
        clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        clf.fit(x_tra, y_tra, batch_size=32, epochs=10, verbose=1, validation_split=0.2)

        #��Ԥ���One-Hot����ת��Ϊnumber�洢
        train_pred= clf.predict(x_tst)
        number1 = reverse_onehot(train_pred)
        second_level_train_set[test_index] =number1

        test_pred=clf.predict(x_test)
        number2 = reverse_onehot(test_pred)
        test_nfolds_sets[:, i]=number2



        #second_level_train_set[test_index] = clf.predict(x_tst)    #���ڴ洢������֤�����ݼ���СΪ 49600һ��
        #test_nfolds_sets[:,i] = clf.predict(x_test)                #��30��������10��Ԥ��ֱ����ÿ���У���Ϊ���ò�ͬѵ����ѵ��10�εõ���ģ�ͣ���Ԥ��10��

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)        #���ڴ洢��test_xԤ������ݼ� ��СΪ1240һ�У�ȡƽ��ֵ��Ľ����
    return second_level_train_set, second_level_test_set



train_sets = []
test_sets = []
for clf in [model2]:
    print('BiGRUģ�Ϳ�ʼ����')
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)
    print('ģ�����н���')

meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)   #������֤��Ԥ����
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)     #��test_x��Ԥ����
#print(meta_train)
#print(meta_test)

#�µ����ݼ�meta_trainά��(49600,2)  train_yά��(49600,10)��Ҫתת��   meta_test(12400,2)   test_y(12400,10)��Ҫתά��
#��������ʹ��5�������㷨�ֱ������ϵ����ݼ�����ѵ����Ԥ��
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


#��train_yά��(49600,10)��Ҫתת��Ϊ��49600����
train_y2=reverse_onehot(train_y)
#test_y(12400,10)��Ҫתά��ת��Ϊ(12400,)
test_y2=reverse_onehot(test_y)



svc_model = SVC()                                   #svc�������Ե�SVM��
svc_model.fit(meta_train,train_y2)
SVC_pred=svc_model.predict(meta_test)



#������֤
from sklearn.model_selection import cross_val_score
print('������֤���:')
print(cross_val_score(svc_model, meta_test, test_y2,cv=3))

#������׼
from sklearn.metrics import classification_report
print('���ȡ��ٻ��ʡ�F1ֵ��')
print(classification_report(test_y2,SVC_pred))







