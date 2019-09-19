import numpy as np
import pandas as pd

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
            array[i][15] = 0
        elif array[i][15] == 'New_Visitor':
            array[i][15] = 1
        elif array[i][15] == 'Other':
            array[i][15] = -1

        if array[i][16] is True:  # Assign Values for Booleans
            array[i][16] = 1
        elif array[i][16] is False:
            array[i][16] = 0
    # array = np.delete(array, [10, 15, 16], 1)
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
X_Pro = np.concatenate((X_tot, X_final), axis=1)
X_norm = np.linalg.norm(X_train, axis=1, keepdims=True)
X_avg = np.mean(X_train, axis=1, keepdims=True)
X_std = np.std(X_train, axis=1, keepdims=True)
X_max = np.max(X_train, axis=1, keepdims=True)
X_min = np.min(X_train, axis=1, keepdims=True)


def normalize(array):
    """Normalizes Data"""
    array = (array - X_avg) / (X_max - X_min)


row_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def normalize_rows(array):
    for i in row_list:
        array[i] = (array[i] - X_avg[i]) / X_std[i]


normalize_rows(X_tot)
normalize_rows(X_train)
normalize_rows(X_test)
normalize_rows(X_final)

# normalize(X_tot)
# normalize(X_train)
# normalize(X_test)
# normalize(X_final)


def get_data():
    d = {"X_tot": X_tot,
         "X_train": X_train,
         "X_test": X_test,
         "X_final": X_final,
         "Y_tot": Y_tot,
         "Y_train": Y_train,
         "Y_test": Y_test}
    return d


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
