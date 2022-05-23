#实验1 BERT与Word2Vec的对比实验
#并列柱状图
import matplotlib.pyplot as plt

name_list = ['Softmax','RF', 'SVM','KNN','GBDT','AdaBoost','Stacking']

num_list= [0.8955,0.8835,0.8708,0.8784,0.8853,0.8942,0.9134]           #ATT-BiLSTM
num_list1= [0.8942,0.8803,0.8671,0.8775,0.8801,0.8881,0.9067]           #ATT-BiGRU
num_list2= [0.8937,0.8819,0.8701,0.8768,0.8821,0.8913,0.9089]           #ATT-BiMGU
num_list3 = [0.9110,0.8893,0.8732,0.8821,0.8869,0.8983,0.9238]           #AGNN
x = list(range(len(num_list1)))
width = 0.4           #宽度
plt.ylim(0.80,0.94)
colorss = ['#DAA520','#c0737a','#03719c','#d6b4fc','#87ae73','grey','#b04e0f']

plt.bar(x, num_list1, width=width, color=colorss,tick_label=name_list)
plt.plot(x, num_list1, color='red',marker='.')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list1, width=width, tick_label=name_list, fc='black')
plt.ylabel('Accuracy')
plt.show()


