#实验1 BERT与Word2Vec的对比实验
#并列柱状图
import matplotlib.pyplot as plt

name_list = ['Softmax','RF', 'SVM','KNN','GBDT','AdaBoost','Stacking']


num_list= [0.9036,0.8897,0.8765,0.8869,0.8895,0.8975,0.9161]           #ATT-BiLSTM
num_list1= [0.9049,0.8927,0.8826,0.8915,0.8963,0.9056,0.9228]           #ATT-BiGRU


num=[]
x = list(range(len(num_list1)))
width = 0.2           #宽度
plt.ylim(0.80,0.94)
colorss = ['grey','black']

plt.bar(x, num_list, width=width, label='Attention-BiLSTM',tick_label=name_list,fc='grey')
# plt.plot(x, num_list1, color='red',marker='.')
for i in range(len(x)):
   x[i] = x[i] + width
plt.bar(x, num_list1, width=width,label='Attention-BiGRU', tick_label=name_list, fc='black')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


