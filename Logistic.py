import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def sigmoid(z):
    """ Compute the sigmoid of z """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """ This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0."""
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


# noinspection PyPep8Naming
def propagate(w, b, X, Y):
    """ Implement the cost function and its gradient for the propagation explained above """
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# noinspection PyPep8Naming
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """ This function optimizes w and b by running a gradient descent algorithm """
    costs = []
    for j in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if j % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and j % 100 == 0:
            print("Cost after iteration %i: %f" % (j, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# noinspection PyPep8Naming
def predict(w, b, X):
    """ Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b) """
    m = X.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    for k in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, k] >= 0.5:
            y_prediction[0, k] = 1

    assert (y_prediction.shape == (1, m))

    return y_prediction


def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """ Builds the logistic regression model by calling the function you've implemented previously """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(x_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, x_train, y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    out = {"costs": costs,
           "Y_prediction_test": y_prediction_test,
           "Y_prediction_train": y_prediction_train,
           "w": w,
           "b": b,
           "learning_rate": learning_rate,
           "num_iterations": num_iterations}

    return out


d = model(X_train, Y_train, X_test, Y_test, num_iterations=5000, learning_rate=0.01, print_cost=True)

# Plotting Cost with Iterations
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# Test Cases Prediction
predict_final = predict(d["w"], d["b"], X_final)
print("1 Count : ", np.count_nonzero(predict_final))

# Upload to File
df = pd.DataFrame(predict_final.T, dtype=int)
df.index += 1
df.to_csv('Data/Predict_reg.csv', sep=',', encoding='utf-8', header=['Revenue'], index_label='ID')


def learning_rate_check():
    """ For Choosing the Correct Learning Rate """
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(X_train, Y_train, X_test, Y_test, num_iterations=1500, learning_rate=i, print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


learning_rate_check()
