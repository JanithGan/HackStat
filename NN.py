import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    array = np.array(array, dtype=np.float)
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
X_avg = np.mean(X_tot, axis=1, keepdims=True)
X_std = np.std(X_tot, axis=1, keepdims=True)
X_max = np.max(X_tot, axis=1, keepdims=True)
X_min = np.min(X_tot, axis=1, keepdims=True)


def normalize(array):
    array = (array - X_avg) / X_std
    return array


X_tot = normalize(X_tot)
X_train = normalize(X_train)
X_test = normalize(X_test)
X_final = normalize(X_final)


def sigmoid(z):
    """ComputeS the sigmoid of z """
    s = 1 / (1 + np.exp(-z))
    return s


def layer_sizes(X, Y):
    """Computes Layer Sizes of NN"""
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """Initializes Parameter for NN Training"""
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def forward_propagation(X, parameters):
    """Forward Propagation to Compute A, Z"""
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters, lambd):
    """Computes Logistic Cost"""
    m = Y.shape[1]  # number of example
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    log_prob = np.log(A2) * Y + np.log(1 - A2) * (1 - Y)
    cost = - np.sum(log_prob) / m

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
    assert (isinstance(cost, float))
    reg_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2))) / (2 * m)

    return cost + reg_cost


def backward_propagation(parameters, cache, X, Y, lambd):
    """Back Propagation to Compute Derivatives"""
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + (lambd * W2) / m
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + (lambd * W1) / m
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """Updates Parameters for Gradient Decent"""
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, learning_rate=1.2, lambd=0.0, print_cost=False):
    """Neural Network Model That Combines Each Step"""

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []

    # Gradient Descent
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters, lambd)

        # Back propagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y, lambd)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters, costs


def predict(parameters, X):
    """Predicts Results for Test Set"""

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = X_new = (A2 >= 0.5)

    return predictions


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


params, cost_table = nn_model(X_train, Y_train, n_h=4, num_iterations=10000, learning_rate=4, lambd=0, print_cost=True)

# Training Set Accuracy
predict_train = predict(params, X_train)
m1 = compute_metrics(Y_train, predict_train)
print('Training Set : ', m1)

# Test Set Accuracy
predict_test = predict(params, X_test)
m2 = compute_metrics(Y_test, predict_test)
print('Test Set : ', m2)

# Total Set Accuracy
predict_tot = predict(params, X_tot)
m3 = compute_metrics(Y_tot, predict_tot)
print('Total Set : ', m3)

# Test Cases Prediction
predict_final = predict(params, X_final)
print(np.count_nonzero(predict_final))

# Upload to File
df = pd.DataFrame(predict_final.T, dtype=int)
df.index += 1
df.to_csv('Data/Predict_nn.csv', sep=',', encoding='utf-8', header=['Revenue'], index_label='ID')


def learning_rate_check():
    """ For Choosing the Correct Learning Rate """
    learning_rates = [4, 1, 0.1]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        p, models[str(i)] = nn_model(X_train, Y_train, n_h=5, num_iterations=1500, learning_rate=i, print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]), label=str(i))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


learning_rate_check()
