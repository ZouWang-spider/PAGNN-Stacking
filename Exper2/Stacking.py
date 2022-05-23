import numpy as np
from sklearn.model_selection import KFold
def get_stacking(clf, x_train, y_train, x_test, n_folds=10):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]     #120   30
    second_level_train_set = np.zeros((train_num,))             #产生1行120列的0矩阵，用于存储预测数据12个为一组
    second_level_test_set = np.zeros((test_num,))               #产生1行30列的0矩阵
    test_nfolds_sets = np.zeros((test_num, n_folds))            #产生30行10列的0矩阵
    kf = KFold(n_splits=n_folds)                          #将训练集分为10分,进行交叉训练

    #i从0-9  train_index有120个  test_index有12个
    # 交叉验证需要将train_x训练集划分10份（每份包含12个数据） 将第一份数据作为test其他部分作为train去训练，然后第二份作为test其他为train训练 依次.........................
    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)     #采用5种算法训练

        second_level_train_set[test_index] = clf.predict(x_tst)    #对每种分别进行预测，存储在每组相应的位置
        test_nfolds_sets[:,i] = clf.predict(x_test)                #对30样本进行10次预测分别存在每列中，因为采用不同训练集训练10次得到的模型，在预测10次

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)        #每行取均值一共得到30组预测结果
    return second_level_train_set, second_level_test_set



#我们这里使用5个分类算法，为了体现stacking的思想，就不加参数了
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

rf_model = RandomForestClassifier()                 #RF随机森林算法
svc_model = SVC()                                   #svc（非线性的SVM）
nb_model=MultinomialNB()
gbdt_model = GradientBoostingClassifier()           #GBDT提升树
adb_model = AdaBoostClassifier()                    #提升算法



#在这里我们使用train_test_split来人为的制造一些数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()                          #使用莺尾花的数据集(150个样本*每个样本4个特征）分为三个种类 0， 1， 2
#print(iris.target.shape)
#data （150,4）      target (150,)
train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)
print(iris.target)
print(train_y.shape)
#数据大小（120,4） （30,4）（120） （30）
train_sets = []
test_sets = []
for clf in [rf_model, svc_model,nb_model, gbdt_model,adb_model]:
    train_set, test_set = get_stacking(clf, train_x, train_y, test_x)
    train_sets.append(train_set)
    test_sets.append(test_set)


print(test_sets)

meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)




#使用决策树作为我们的次级分类器
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()                         #决策树算法（CART决策树）
dt_model.fit(meta_train, train_y)
df_predict = dt_model.predict(meta_test)
print(df_predict)
