import numpy as np

training_inputs = np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1]])
training_outputs = np.array([[0, 1, 0, 1]]).T

print('Training Inputs:\n', training_inputs)
print('Training Outputs:\n', training_outputs)


def sigmoid(x):

    """
    Applies sigmoid function to the input
    input: x
    output: 1/1+exponent of negative x

    """
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):

    """
    Applies sigmoid function to the input
    input: x
    output: sigmoid(x)(1 - sigmoid(x)) where sigmoid = x/1+exp(-x)

    """
    sig_value = 1/(1 + np.exp(-x))
    derivative = sig_value * (1 - sig_value)
    return derivative
    # return x * (1 - x)


initial_weights = 2 * np.random.random((3, 1)) - 1
print('Initial Weight (Random):\n', initial_weights)

nn_weights = initial_weights

for i in range(1000000):
    prediction = sigmoid(np.dot(training_inputs, nn_weights))
    error = training_outputs - prediction
    # print('Error:\n', error)
    adjusted = error * sigmoid_derivative(prediction)
    # print('Adjusted:\n', error)
    nn_weights += np.dot(training_inputs.T, adjusted)


print('Prediction:\n', prediction)
print('NN weight:\n', nn_weights)





