import numpy as np
import pandas as pd
import sklearn.linear_model

# This is the Code for Built-in sklearn Linear Regression

raw_data = pd.read_csv('Data/Trainset.csv')
raw_data.dropna(axis=0, how='any', inplace=True)
raw_data = np.array(raw_data)

months = []


def process_data(array):
    m_init = array.shape[0]
    for i in range(m_init):
        mon = array[i][10]
        if mon not in months:
            months.append(mon)
        array[i][10] = months.index(mon)

        if array[i][15] == 'Returning_Visitor':
            array[i][15] = 2
        elif array[i][15] == 'New_Visitor':
            array[i][15] = 1
        elif array[i][15] == 'Other':
            array[i][15] = 0

        if array[i][16] is True:
            array[i][16] = 1
        elif array[i][16] is False:
            array[i][16] = 0
    array = np.array(list(array), dtype=np.float)
    return array


data = process_data(raw_data)
print(data.shape)
X_tot = data[:, :-1]
Y_tot = data[:, -1]
print(data.shape)

X_train = data[:, :-1].T
Y_train = data[:, -1].T

X_norm = np.linalg.norm(X_train, axis=1, keepdims=True)
X_train = X_train / X_norm

# train = data[:7500].T
# test = data[7500:].T
#
# X_train = train[:-1]
# n = X_train.shape[0]
# m_train = X_train.shape[1]
# Y_train = train[-1]  # .reshape(1, m_train)
#
# X_test = test[:-1]
# m_test = X_test.sha pe[1]
# Y_test = test[-1].reshape(1, m_test)
#
# X_norm = np.linalg.norm(X_train, axis=1, keepdims=True)
# X_train = X_train / X_norm
# X_test = X_test / X_norm

clf = sklearn.linear_model.LogisticRegressionCV(max_iter=10000)
clf.fit(X_train.T, Y_train.T)
s = clf.predict(X_train.T)

print("train accuracy: {} %".format(100 - np.mean(np.abs(s - Y_train)) * 100))
print(np.count_nonzero(s))

# t = clf.predict(X_test.T)
# print("test accuracy: {} %".format(100 - np.mean(np.abs(t - Y_test)) * 100))
# print(np.count_nonzero(t))

test_data = pd.read_csv('Data/xtest.csv')
test_data.dropna(axis=0, how='any', inplace=True)
test_data = np.array(test_data)

test_data = test_data[:, 1:]
test_cases = process_data(test_data).T
X_Test_final = test_cases / X_norm
y_pred = clf.predict(X_Test_final.T)
print(np.count_nonzero(y_pred))

df = pd.DataFrame(y_pred.T, dtype=int)
df.index += 1
df.to_csv('Data/MyPredict1.csv', sep=',', encoding='utf-8', header=['Revenue'], index_label='ID')
