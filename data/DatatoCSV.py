import os
import sys
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd


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
        fullname=class_path+file_path                         #文件的全路径
        data=readfile(fullname).decode('gbk').encode('utf8').decode()         #decode()       #decode('gbk').encode('utf8').decode()                      #读取文件内容 #bytes→str需要decode()解码操作将字节转化为文字
        data1=data.replace('\n','')     #删除换行和多余的空格
        data2=data1.replace('\r','')
        data3=data2.replace(' ','')
        data4=data3.strip()
        str=data4
        #为文件的内容分词并将分词存入train_corpus列表(每一个元素存储一篇文档的内容）
        #seg_list = jieba.cut(str, cut_all=True)               #分词
        #train_corpus.append(" ".join(seg_list))
        #train_corpus.append(" ".join(str))

        train_corpus.append(str)
        train_label.append(mydir)
        data=pd.DataFrame(train_corpus,index=train_label)
        print(data)
        print(file_path )
        data.to_csv('data5.csv')
        #print(file_path)

print('分词结束')
