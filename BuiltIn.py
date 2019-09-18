import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Reading Train Set Data File
raw_d1 = pd.read_csv('Data/Trainset.csv')
raw_d1.dropna(axis=0, how='any', inplace=True)

# Reading Test Case Set Data File
raw_d2 = pd.read_csv('Data/xtest.csv')
raw_d2.dropna(axis=0, how='any', inplace=True)

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
        array[i][10] = months.index(mon) + 1

        if array[i][15] == 'Returning_Visitor':  # Assign Values for Visitor Types
            array[i][15] = 3
        elif array[i][15] == 'New_Visitor':
            array[i][15] = 2
        elif array[i][15] == 'Other':
            array[i][15] = 1

        if array[i][16] is True:  # Assign Values for Booleans
            array[i][16] = 2
        elif array[i][16] is False:
            array[i][16] = 1

    array = np.array(array, dtype=np.float)
    return array


# Processing Data
data = process_data(train_data)
test = process_data(test_data)

# Total Data
X_tot = data[:, :-1].T
m_tot = X_tot.shape[1]
Y_tot = data[:, -1].reshape(1, m_tot)

div_const = 7500  # 500 Multiples Only

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
X = np.concatenate((X_tot, X_final),axis=1)
print(X.shape)
X_norm = np.linalg.norm(X, axis=1, keepdims=True)
X_avg = np.mean(X, axis=1, keepdims=True)
X_std = np.std(X, axis=1, keepdims=True)
X_max = np.max(X, axis=1, keepdims=True)
X_min = np.min(X, axis=1, keepdims=True)


def normalize(array):
    """Normalizes Data"""
    array = (array - X_avg) / X_std
    return array


X_tot = normalize(X_tot)
X_train = normalize(X_train)
X_test = normalize(X_test)
X_final = normalize(X_final)


def compute_metrics(y, predict_y):
    TP = np.sum(np.logical_and(predict_y == 1, y == 1))
    TN = np.sum(np.logical_and(predict_y == 0, y == 0))
    FP = np.sum(np.logical_and(predict_y == 1, y == 0))
    FN = np.sum(np.logical_and(predict_y == 0, y == 1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) * 100.0 / (TP + TN + FP + FN)
    f_score = (2 * precision * recall) * 100.0 / (precision + recall)

    metrics = {"Accuracy %": round(accuracy, 5),
               "F_score %": round(f_score, 5),
               "Precision": round(precision, 5),
               "Recall": round(recall, 5)}
    return metrics


# Training Linear Regression Using SkLearn
# clf = sklearn.linear_model.LogisticRegressionCV(max_iter=10000)
clf = RandomForestClassifier(n_estimators=500)
# clf = ComplementNB()
# clf = MultinomialNB()
clf.fit(X_train.T, np.squeeze(Y_train.T))

# Prediction
predict_train = clf.predict(X_train.T)
predict_test = clf.predict(X_test.T)
predict_tot = clf.predict(X_tot.T)
predict_final = clf.predict(X_final.T)

# Training Set Accuracy
m1 = compute_metrics(Y_train, predict_train)
print('Training Set : ', m1)

# Test Set Accuracy
m2 = compute_metrics(Y_test, predict_test)
print('Test Set : ', m2)

# Total Set Accuracy
m3 = compute_metrics(Y_tot, predict_tot)
print('Total Set : ', m3)

# Test Cases Prediction
print(np.count_nonzero(predict_final))

# Upload to File
df = pd.DataFrame(predict_final.T, dtype=int)
df.index += 1
df.to_csv('Data/Predict_skl.csv', sep=',', encoding='utf-8', header=['Revenue'], index_label='ID')
