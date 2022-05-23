import numpy as np
import pandas as pd

#处理标签
filepath= 'online_shopping_10_cats.csv'

df=pd.read_csv(filepath, error_bad_lines=False)
# 标签及词汇
labels, content = list(df['label'].unique()), list(df['content'].unique())

df['id'] = df['label'].factorize()[0]
cat_id_df = df[['label', 'id']].drop_duplicates().sort_values('id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['id', 'label']].values)
id = df['id'].values
df.to_csv('online2.csv')



y = pd.get_dummies(df['id']).values
print(y.shape)

#将One-Hot编码反转为id
def reverse_onehot(onehot_data):
    # onehot_data assumed to be channel last
    data_copy = np.zeros(onehot_data.shape[:-1])
    for c in range(onehot_data.shape[-1]):
        img_c = onehot_data[..., c]
        data_copy[img_c == 1] = c
    return data_copy

number=reverse_onehot(y)
#print(number)


second_level_train_set = np.zeros((62000,))
test_nfolds_sets = np.zeros((62000, 10))
second_level_test_set = np.zeros((62000,))

for i in range(10):
   number2 = reverse_onehot(y)
   test_nfolds_sets[:, i]=number2

second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
print(second_level_test_set)

test_sets = np.zeros((5, 5))            #产生12400行5列的0矩阵
test_sets[:, 0] = [1,2,3,4,5]
test_sets[:, 1] = [2,3,4,5,6]
test_sets[:, 2] = [3,4,5,6,7]
test_sets[:, 3] = [4,5,6,7,8]
test_sets[:, 4] = [5,6,7,8,9]
print(test_sets)