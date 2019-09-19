import matplotlib.pyplot as plt
from Deep_Utils import *
from Data_Process import *


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, lambd=0.0, num_iterations=3000, print_cost=False):
    """Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID."""

    costs = []  # keep track of cost

    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y, layers_dims, parameters, lambd)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, lambd)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# Define Size for Each Layer - Change This
dims = [17, 10, 5, 1]

# Training the Model - Change This
params = L_layer_model(X_train, Y_train, dims, num_iterations=5000, lambd=0, learning_rate=3, print_cost=True)

predict_train = predict(X_train, params)
predict_test = predict(X_test, params)
predict_tot = predict(X_tot, params)
predict_final = predict(X_final, params)

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
df.to_csv('Data/Predict_dnn.csv', sep=',', encoding='utf-8', header=['Revenue'], index_label='ID')
