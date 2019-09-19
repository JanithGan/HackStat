import matplotlib.pyplot as plt
from Data import *


def sigmoid(z):
    """Computes the sigmoid of z"""
    s = 1 / (1 + np.exp(-z))
    return s


def ReLU(x):
    return x * (x > 0)


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
    # A1 = ReLU(Z1)
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
    epsilon = 0
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

    # dA1 = np.dot(W2.T, dZ2)  # For ReLU
    # dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))   # For tanh
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


def nn_model(X, Y, n_h, X_t=None, Y_t=None, num_iterations=10000, learning_rate=4, lambd=0.0, print_cost=False):
    """Neural Network Model That Combines Each Step"""

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []
    train_cost = []

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
            if not (X_t is None or Y_t is None):
                pre = predict(parameters, X_t)
                t_cost = compute_cost(pre["A2"], Y_t, parameters, lambd)
                train_cost.append(t_cost)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    d = {"params": parameters,
         "costs": costs,
         "t_costs": train_cost}

    return d


def predict(parameters, X):
    """Predicts Results for Test Set"""

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 >= 0.5)

    d = {"P": predictions,
         "A2": A2}

    return d


# Training the Model
out = nn_model(X_train, Y_train, n_h=10, num_iterations=10000, learning_rate=4, lambd=0, print_cost=True)

# Predicting Values
predict_train = predict(out["params"], X_train)
predict_test = predict(out["params"], X_test)
predict_tot = predict(out["params"], X_tot)
# predict_final = predict(out["params"], X_final)

# Training Set Accuracy
m1 = compute_metrics(Y_train, predict_train["P"])
print('Training Set : ', m1)

# Test Set Accuracy
m2 = compute_metrics(Y_test, predict_test["P"])
print('Test Set : ', m2)

# Total Set Accuracy
m3 = compute_metrics(Y_tot, predict_tot["P"])
print('Total Set : ', m3)
#
# # Test Cases Prediction
# predict_f = predict_final["P"]
# print(np.count_nonzero(predict_f))
#
# # Upload to File
# df = pd.DataFrame(predict_f.T, dtype=int)
# df.index += 1
# df.to_csv('Data/Predict_nn.csv', sep=',', encoding='utf-8', header=['Revenue'], index_label='ID')


# Curve Plotting
# --------------

def plot_learning_curves(n_h=5, n_i=1500, l_r=4, lambd=0.0):
    """Plots Learning Curves"""
    cost1, cost2 = [], []
    nums = list(range(500, div_const + 500, 500))
    print("Plotting Learning Curves... ")
    for m in nums:
        D = nn_model(X_train[:, :m], Y_train[:, :m], n_h=n_h, X_t=X_test, Y_t=Y_test, num_iterations=n_i,
                     learning_rate=l_r, lambd=lambd, print_cost=False)
        cost1.append(D["costs"][-1])
        cost2.append(D["t_costs"][-1])

    z = 1
    plt.plot(nums[z:], cost1[z:], label="Train")
    plt.plot(nums[z:], cost2[z:], label="Test")

    plt.ylabel('Cost')
    plt.xlabel('Training Set Size')
    plt.title('Learning Curves')

    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def plot_cost_curves(n_h=5, n_i=10000, l_r=4, lambd=0.0):
    print("Plotting Cost Curve... ")

    D = nn_model(X_train, Y_train, n_h=n_h, X_t=X_test, Y_t=Y_test, num_iterations=n_i, learning_rate=l_r, lambd=lambd,
                 print_cost=False)

    plt.plot(np.squeeze(D["costs"][1:]), label="Train")
    plt.plot(np.squeeze(D["t_costs"][1:]), label="Test")

    plt.ylabel('Cost')
    plt.xlabel('Iterations (hundreds)')
    plt.title('Curve for Cost vs Iterations')

    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def learning_rate_check():
    """For Choosing the Correct Learning Rate"""
    learning_rates = [4, 1, 0.1]
    models = {}
    print("Plotting Learning Rate vs Cost ... ")
    for i in learning_rates:
        D = nn_model(X_train, Y_train, n_h=5, num_iterations=1500, learning_rate=i, lambd=0.0, print_cost=False)
        models[str(i)] = D["costs"]

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)][1:]), label=str(i))

    plt.ylabel('Cost')
    plt.xlabel('Iterations (hundreds)')
    plt.title('Learning Rate Check')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def lambda_check():
    """For Choosing the Correct Lambda"""
    lambdas = [0, 0.1, 1]
    models = {}
    print("Plotting Lambdas vs Cost ... ")
    for i in lambdas:
        D = nn_model(X_train, Y_train, n_h=5, num_iterations=1500, learning_rate=4, lambd=i, print_cost=False)
        models[str(i)] = D["costs"]

    for i in lambdas:
        plt.plot(np.squeeze(models[str(i)][1:]), label=str(i))

    plt.ylabel('Cost')
    plt.xlabel('Iterations (hundreds)')
    plt.title('Lambda Check')

    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


# plot_learning_curves(n_h=8)
# plot_cost_curves()
# learning_rate_check()
# lambda_check()
