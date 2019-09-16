import numpy as np
import pandas as pd
import sklearn.linear_model

# Reading Train Set Data File
raw_d1 = pd.read_csv('Data/Trainset.csv')
raw_d1.dropna(axis=0, how='any', inplace=True)

# Reading Test Case Set Data File
raw_d2 = pd.read_csv('Data/xtest.csv')
raw_d1.dropna(axis=0, how='any', inplace=True)

# Convert Data to ndArrays
train_data = np.array(raw_d1)
test_data = np.array(raw_d2)[:, 1:]

months = []


def process_data(array):
    """ This function pre-processes raw data"""
    m_init = array.shape[0]
    for i in range(m_init):
        mon = array[i][10]  # Assign Values for Months
        if mon not in months:
            months.append(mon)
        array[i][10] = months.index(mon)

        if array[i][15] == 'Returning_Visitor':  # Assign Values for Visitor Types
            array[i][15] = 2
        elif array[i][15] == 'New_Visitor':
            array[i][15] = 1
        elif array[i][15] == 'Other':
            array[i][15] = 0

        if array[i][16] is True:  # Assign Values for Booleans
            array[i][16] = 1
        elif array[i][16] is False:
            array[i][16] = 0

    array = np.array(list(array), dtype=np.float)
    return array


# Processing Data
data = process_data(train_data)
test = process_data(test_data)

# Total Data
X_tot = data[:, :-1].T
m_tot = X_tot.shape[1]
Y_tot = data[:, -1].reshape(1, m_tot)

div_const = 7500

# Training Set Data
X_train = data[:div_const, :-1].T
m_train = X_train.shape[1]
Y_train = data[:div_const, -1].reshape(1, m_train)

# Test Set Data
X_test = data[div_const:, :-1].T
m_test = X_test.shape[1]
Y_test = data[div_const:, -1].reshape(1, m_test)

# Test Cases Data
X_final = test.T
m_final = X_final[1]

# Normalizing Data
X_norm = np.linalg.norm(X_tot, axis=1, keepdims=True)
X_tot = X_tot / X_norm
X_train = X_train / X_norm
X_test = X_test / X_norm
X_final = X_final / X_norm

# Training Linear Regression Using SkLearn
clf = sklearn.linear_model.LogisticRegressionCV(max_iter=10000)
clf.fit(X_tot.T, Y_tot.T)

# Training Set Accuracy
predict_train = clf.predict(X_train.T)
print("train accuracy: {} %".format(100 - np.mean(np.abs(predict_train - Y_train)) * 100))
print("1 Count : ", np.count_nonzero(predict_train))

# Test Set Accuracy
predict_test = clf.predict(X_test.T)
print("test accuracy: {} %".format(100 - np.mean(np.abs(predict_test - Y_test)) * 100))
print("1 Count : ", np.count_nonzero(predict_test))

# Test Cases Prediction
predict_final = clf.predict(X_final.T)
print("1 Count : ", np.count_nonzero(predict_final))

# Upload to File
df = pd.DataFrame(predict_final.T, dtype=int)
df.index += 1
df.to_csv('Data/Predict_skl.csv', sep=',', encoding='utf-8', header=['Revenue'], index_label='ID')
