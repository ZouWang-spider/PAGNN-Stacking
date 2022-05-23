# -*-coding: UTF-8 -*-
# @Time:2019/9/1016:11
# @author superxjz
# @func
import pandas as pd
import os
import numpy as np
from nltk.corpus import stopwords
import re
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import accuracy_score
import nltk
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import gensim

#第一步---对原始数加工成表格一个csv文件
def dataToexcel():
    #创建数据的形式
    data = pd.DataFrame()
    #对文件进行遍历
    for s in ("test","train"):
        for l in ("pos","neg"):
            path = "C:/Users/Administrator/Desktop/IMDB数据集/aclImdb/%s/%s" % (s, l)
            #遍历文本文档
            for file in os.listdir(path):
                with open(os.path.join(path,file),"r",encoding="utf-8")as fr:
                    #将文本中的句子读出来
                    sentence = fr.read()
                    if l=="pos":
                        data=data.append([[sentence,1]],ignore_index=True)
                    if l=="neg":
                        data = data.append([[sentence, 0]], ignore_index=True)

    #给表格设置列名
    data.columns=["review","label"]

    #将数据的顺序打乱
    np.random.seed(0)
    data = data.reindex(np.random.permutation(data.index))

    #将数据保存成csv文件
    data.to_csv("imdb.csv")
    filePath="imdb.csv"
    #将文件的路径返回
    return filePath


#第二步---将第一步生成csv文件中所有的评论进行数据的清洗操作
def clearText(text):
    #读取数据
    #data = pd.read_csv(filePath,encoding="utf-8")
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    # ','.join('abc')-----'a,b,c'
    return " ".join(words)

#第三步进行特征向量的建立
def bulidVec(text,model):
    #定义一个数组
    vec = []
    wordList = text.split()
    for word in wordList:
        word = word.replace("\n", " ")
        try:
            vec.append(model[word])
        except KeyError:
            continue
    return vec



#测试
#dataToexcel()
if __name__=="__main__":
    #第一步
    # nltk.download()
    filePath = dataToexcel()

    #第二步
    data = pd.read_csv(filePath, encoding="utf-8")
    eng_stopwords = set(stopwords.words("english"))
    data["clear_review"] = data.review.apply(clearText)

    #第三步
    vectorizer = CountVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data.clear_review).toarray()
    # data["array"] = data.clear_review.apply(bulidVec)
    # X = data["array"].values
    Y = data["label"].values

    #对影评数据进行分析
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.30, shuffle=True)
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr', probability=True)
    # clf = RandomForestClassifier(n_estimators=100)
    clf = GaussianNB()
    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("源标签数据",Y)
    print("预测标签数据",clf.predict(X))

    print("混淆矩阵",confusion_matrix(data.label, clf.predict(X)))
    print("精确度", precision_score(Y, clf.predict(X)))
    print("精确度2",precision_score(y_test, clf.predict(X_test)))


