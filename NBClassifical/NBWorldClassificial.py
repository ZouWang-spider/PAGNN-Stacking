#coding:utf-8
#数据集为复旦大学语料的数据集train_corpus
#(art,compoter,economic,education,environment,medical,military,politics,sport,traffic)每个包含有250篇左右的txt文件
import os
import sys
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#读取文件
def readfile(path):
    fp=open(path,'rb')
    content=fp.read()
    fp.close()
    return content

#定义一个存放训练集分词后的了内容
train_corpus=[]
#标签存放
train_label=[]

#语料路径
corpus_path='E:/Pythonproject/LSTM&Stacking/fudan/train/'
#获取未分词的语料所有子目录列表cate_list
#['art','computer','econmoic','education','environment','medical','military','politics','sports','traffic']
cate_list=os.listdir(corpus_path)

#获取子目录下的所用文件
for mydir in cate_list:
    class_path=corpus_path+mydir+'/'
    file_list=os.listdir(class_path)


    for file_path in file_list:
        fullname=class_path+file_path                    #文件的全路径
        data=readfile(fullname).decode()                      #读取文件内容 #bytes→str需要decode()解码操作将字节转化为文字
        data1=data.replace('\n','')     #删除换行和多余的空格
        d=data1.strip()
        str=d
        #为文件的内容分词并将分词存入train_corpus列表(每一个元素存储一篇文档的内容）
        seg_list = jieba.cut(str, cut_all=True)
        #print(" ".join(seg_list))
        train_corpus.append(" ".join(seg_list))
        train_label.append(mydir)

print('分词结束')



#读取停用词
stopword_path='stopword'
stoplst=readfile(stopword_path).splitlines()

#TFIDF 将文本中的词语转换为词频举证CountVectorizer
vectorizer=CountVectorizer(stop_words=stoplst,max_df=0.5)
transformer=TfidfTransformer()

#第一个fit_transformer用来计算TFIDF，第二个fit_transformer用来将文本装换为词频矩阵
tfidf=transformer.fit_transform(vectorizer.fit_transform(train_corpus))

#查看所有文档形成的TF-IDF矩阵，每一行表示一篇文档每一列表示一个词
weight=tfidf.toarray()
print(weight)
print('生成TF-TDF矩阵结束')

#导入样本划分函数
from sklearn.model_selection import train_test_split
#按3:7划分为训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(tfidf,train_label,test_size=0.3)






from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
#训练模型
byes_clf=MultinomialNB(alpha=0.01,fit_prior=True)
byes_clf.fit(x_train,y_train)

#模型的存储
import joblib
#save model
joblib.dump(byes_clf, 'byes_clf.pkl')
#load model
byes_clf = joblib.load('byes_clf.pkl')

'''''
import pickle
#save model
f = open('byes_clf.pickle','wb')
pickle.dump(byes_clf,f)
f.close()
#load model
f = open('byes_clf.pickle','rb')
byes_clf = pickle.load(f)
f.close()
'''


#预测
y_pre=byes_clf.predict(x_test)

#模型指标衡量
print('交叉验证的结果')
from sklearn.model_selection import cross_val_score
print(cross_val_score(byes_clf,x_test,y_test,cv=5))

print('precision、recall、F1值')
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pre,target_names=None))

print('混淆矩阵')
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pre))




#