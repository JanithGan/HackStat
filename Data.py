import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import Normalizer

# Reading Train Set Data File
raw_d1 = pd.read_csv('Data/Preprocessed/P_Train.csv')
raw_d1.dropna(axis=0, how='any', inplace=True)

# Reading Test Case Set Data File
raw_d2 = pd.read_csv('Data/Preprocessed/P_Test.csv')
raw_d2.dropna(axis=0, how='any', inplace=True)

data = np.array(raw_d1)
np.random.shuffle(data)
test = np.array(raw_d2)

# Total Training Data
X_data = data[:, :-1]

k_model = SelectKBest(mutual_info_classif, k=25)
k_model.fit(X_data, data[:, -1])
X_data = k_model.transform(X_data)

X_tot = X_data.T
m_tot = X_tot.shape[1]
Y_tot = data[:, -1].reshape(1, m_tot)
print(X_tot.shape)

div_const = 8500  # 500 Multiples Only

# Training Set Data
X_train = X_tot[:, :div_const]
m_train = X_train.shape[1]
Y_train = data[:div_const, -1].reshape(1, m_train)

# Test Set Data
X_test = X_tot[:, div_const:]
m_test = X_test.shape[1]
Y_test = data[div_const:, -1].reshape(1, m_test)

# Test Cases Data
X_final = k_model.transform(test).T
# X_final = test.T
m_final = X_final[1]

# Normalizing Data
X_Pro = np.concatenate((X_tot, X_final), axis=1)

normalize = Normalizer(norm='max')
normalize.fit(X_tot)

X_tot = normalize.transform(X_tot)
X_train = normalize.transform(X_train)
X_test = normalize.transform(X_test)
X_final = normalize.transform(X_final)


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
