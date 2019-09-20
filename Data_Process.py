import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import chi2

# Reading Train Set Data File
raw_d1 = pd.read_csv('Data/Trainset.csv')
raw_d1.dropna(axis=0, how='any', inplace=True)

# Reading Test Case Set Data File
raw_d2 = pd.read_csv('Data/xtest.csv')
raw_d2.dropna(axis=0, how='any', inplace=True)

# Convert Data to ndArrays
train_data = np.array(raw_d1)
np.random.shuffle(train_data)
test_data = np.array(raw_d2)[:, 1:]

months = list(np.unique(np.concatenate((train_data[:, 10], test_data[:, 10]), axis=0)))
months.sort()


def process_data(array):
    """ This function pre-processes raw data"""
    m_init = array.shape[0]
    for i in range(m_init):
        mon = array[i][10]  # Assign Values for Months
        array[i][10] = months.index(mon) + 1

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
    # array = np.delete(array, [10, 15, 16], 1)
    array = np.array(array, dtype=np.float)

    return array


# Processing Data
data = process_data(train_data)
data = data.T
data = data[:, data[1] != -1].T
test = process_data(test_data)

# Total Training Data
X_data = data[:, :-1]

k_model = SelectKBest(mutual_info_classif, k=15)
k_model.fit(X_data, data[:, -1])
X_tot = k_model.transform(X_data)
X_tot = X_tot.T

# X_tot = X_data.T  # Without Feature Removal
m_tot = X_tot.shape[1]
Y_tot = data[:, -1].reshape(1, m_tot)
print(X_tot.shape)

div_const = 9000  # 500 Multiples Only

# Training Set Data
X_train = X_tot[:, :div_const]
m_train = X_train.shape[1]
Y_train = data[:div_const, -1].reshape(1, m_train)

# Test Set Data
X_test = X_tot[:, div_const:]
m_test = X_test.shape[1]
Y_test = data[div_const:, -1].reshape(1, m_test)

# Test Cases Data
X_final = k_model.transform(test)
X_final = X_final.T
m_final = X_final[1]

print(X_final.shape)

# Normalizing Data
X_Pro = np.concatenate((X_tot, X_final), axis=1)
X_norm = np.linalg.norm(X_tot, axis=1, keepdims=True)
X_avg = np.mean(X_tot, axis=1, keepdims=True)
X_std = np.std(X_tot, axis=1, keepdims=True)
X_max = np.max(X_tot, axis=1, keepdims=True)
X_min = np.min(X_tot, axis=1, keepdims=True)


def normalize(array):
    """Normalizes Data"""
    array = (array - X_avg) / (X_max - X_min)
    return array


row_list = list(range(9))


def normalize_rows(array):
    for i in row_list:
        array[i] = (array[i] - X_avg[i]) / X_std[i]
    return array


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
