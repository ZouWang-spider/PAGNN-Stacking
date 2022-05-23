import pandas as pd
import os
import numpy as np

#第一步---对原始数加工成表格一个csv文件

#创建数据的形式
data = pd.DataFrame()
#对文件进行遍历
# for s in ("test","train"):
# for l in ("pos","neg"):
s='test'
l='neg'
path = "C:/Users/Administrator/Desktop/IMDB数据集/aclImdb/%s/%s" % (s, l)
for file in os.listdir(path):
                with open(os.path.join(path,file),"r",encoding="utf-8")as fr:
                    #将文本中的句子读出来
                    sentence = fr.read()
                    if l=="pos":
                        data=data.append([[1,sentence]],ignore_index=True)
                    if l=="neg":
                        data = data.append([[0,sentence]], ignore_index=True)

#给表格设置列名
data.columns=["label","content"]

#将数据的顺序打乱
np.random.seed(0)
data = data.reindex(np.random.permutation(data.index))

#将数据保存成csv文件
data.to_csv("li4.csv")
#将文件的路径返回

#使用时将第一列删除
import pandas as pd
df =pd.read_csv('imdb.csv')
df1=df.drop(columns=['id'])
print(df1)            #直接使用df1



