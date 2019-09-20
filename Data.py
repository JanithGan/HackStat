import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Reading Train Set Data File
raw_d1 = pd.read_csv('Data/Preprocessed/P_Train.csv')
raw_d1.dropna(axis=0, how='any', inplace=True)

# Reading Test Case Set Data File
raw_d2 = pd.read_csv('Data/Preprocessed/P_Test.csv')
raw_d2.dropna(axis=0, how='any', inplace=True)

data = np.array(raw_d1)
np.random.shuffle(data)
test = np.array(raw_d2)

# Total Data
X_tot = data[:, :-1].T
m_tot = X_tot.shape[1]
Y_tot = data[:, -1].reshape(1, m_tot)

div_const = 8500  # 500 Multiples Only

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

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, X_final.shape)

# Normalizing Data
X_Pro = np.concatenate((X_tot, X_final), axis=1)
X_norm = np.linalg.norm(X_Pro, axis=1, keepdims=True)
X_avg = np.mean(X_Pro, axis=1, keepdims=True)
X_std = np.std(X_Pro, axis=1, keepdims=True)
X_max = np.max(X_Pro, axis=1, keepdims=True)
X_min = np.min(X_Pro, axis=1, keepdims=True)


def normalize(array):
    """Normalizes Data"""
    array = (array - X_avg) / X_std
    return array


row_list = list(range(X_train.shape[0]))


# def normalize_rows(array):
#     for i in row_list:
#         # array[i] = (array[i] - X_avg[i]) / X_std[i]
#         array[i] = nz(array[i], norm='l1')
#     return array


# X_tot = normalize_rows(X_tot)
# X_train = normalize_rows(X_train)
# X_test = normalize_rows(X_test)
# X_final = normalize_rows(X_final)

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
