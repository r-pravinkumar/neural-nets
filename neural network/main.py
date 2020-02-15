import numpy as np


class MyNeuralNetwork:

    def __init__(self, random_state=40):
        np.random.seed(random_state)
        self.nn_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):

        """
        Applies sigmoid function to the input
        input: x
        output: 1/1+exponent of negative x

        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):

        """
        Applies sigmoid function to the input
        input: x
        output: sigmoid(x)(1 - sigmoid(x)) where sigmoid = x/1+exp(-x)

        """
        sig_value = 1 / (1 + np.exp(-x))
        derivative = sig_value * (1 - sig_value)
        return derivative

    def fit(self, train_input, train_output, iteration=100):
        """
        Trains the network/perceptron with the data
        input:
            train_input : training input
            train_output : training output
            iteration: iteration value for the training
        output:
            none, we ll use the nn_weights after training for prediction

        """
        for i in range(iteration):
            node_out = self.sigmoid(np.dot(train_input, self.nn_weights))
            self.nn_weights += np.dot(train_input.T, (train_output - node_out) * self.sigmoid_derivative(node_out))

    def predict(self, valid_input):
        """
        input :
            valid_input : validation input/test input
        output :
            return the prediction output
        """
        return self.sigmoid(np.dot(valid_input, self.nn_weights))


if __name__ == '__main__':

    # let relation b/w input and output would be (A and B or C)
    training_inputs = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1]])
    training_outputs = np.array([[1, 0, 0, 1]]).T

    model = MyNeuralNetwork(random_state=1)
    model.fit(training_inputs, training_outputs, iteration=1000)
    prediction = model.predict([[0, 0, 1], [0, 0, 0]])

    print('Prediction:', prediction)







