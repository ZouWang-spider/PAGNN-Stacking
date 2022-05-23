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


#存放测试集文档名称
test_name=[]
# 定义一个存放测试集分词后的了内容
test_corpus = []

# 语料路径
test_corpus_path = 'E:/Pythonproject/LSTM&Stacking/fudan/test/'
#获取未分词的语料所有子目录列表cate_list,复旦大学文本数据集包含20类别，其中train_共有9804篇文档，test共有9832篇文档
#只取10种类别，每个类别存放250篇左右文档
# ['art','computer','econmoic','education','environment','medical','military','politics','sports','traffic']
test_file_list = os.listdir(test_corpus_path)


# 获取子目录下的所用文件
for mydir in test_file_list:
    class_path=test_corpus_path+mydir+'/'
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
        test_corpus.append(" ".join(seg_list))
        test_name.append(mydir)

print('分词结束')


# 读取停用词
stopword_path = 'stopword'
stoplst = readfile(stopword_path).splitlines()

# TFIDF 将文本中的词语转换为词频举证CountVectorizer
test_vectorizer = CountVectorizer(stop_words=stoplst, max_df=0.5)
transformer = TfidfTransformer()

# 第一个fit_transformer用来计算TFIDF，第二个fit_transformer用来将文本装换为词频矩阵
test_tfidf = transformer.fit_transform(test_vectorizer.fit_transform(test_corpus))

#加载模型
import joblib
byes_clf = joblib.load('byes_clf.pkl')
#进行新数据的预测
predcited=byes_clf.predict(test_tfidf)

#将预测的结果和文档名称对应起来形成DataFrame
import pandas as pd
result=pd.DataFrame({'filename':test_name,'ctae_predicted':predcited})

#将预测结果写入CSV文件并保存
result.to_csv('',encoding='utf-8',index=False)