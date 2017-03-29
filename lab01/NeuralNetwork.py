import time
import random
import numpy as np
from utils import *
from transfer_functions import *


class NeuralNetwork(object):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, iterations=50, learning_rate = 0.1):
        """
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        iterations: how many iterations
        learning_rate: initial learning rate
        """
        # initialize parameters
        self.iterations = iterations        # iterations
        self.learning_rate = learning_rate

        # initialize arrays
        self.input = input_layer_size + 1   # +1 for the bias node in the input layer
        self.hidden = hidden_layer_size + 1 # +1 for the bias node in the hidden layer
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden = np.ones(self.hidden)
        self.a_output = np.ones(self.output)

        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input**(1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden - 1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)


    def weights_initialisation(self,wi,wo):
        self.W_input_to_hidden=wi # weights between input and hidden layers
        self.W_hidden_to_output=wo # weights between hidden and output layers


    def feed_forward(self, inputs):
        # Compute input activations
        self.a_input = np.array(inputs)
        if self.a_input.size < self.input:
            self.a_input = np.append(self.a_input, 1)
        self.a_input = np.atleast_2d(self.a_input)

        # Compute hidden activations
        self.a_hidden = self.a_input.dot(self.W_input_to_hidden)
        self.o_hidden = sigmoid(self.a_hidden)
        if len(self.o_hidden) < self.hidden:
            self.o_hidden = np.append(self.o_hidden, 1)
        self.o_hidden = np.atleast_2d(self.o_hidden)

        # Compute output activations
        self.a_output = self.o_hidden.dot(self.W_hidden_to_output)
        self.output = sigmoid(self.a_output)
        return self.output


    def back_propagate(self, targets):
        # Calculate error terms for output
        dEdu2 = np.multiply(self.output - targets, np.multiply(self.output, 1 - self.output))
        dEdu2 = np.atleast_2d(dEdu2)

        # Calculate error terms for hidden
        dEdu1 = np.multiply(dEdu2.dot(self.W_hidden_to_output.T), np.multiply(self.o_hidden, 1 - self.o_hidden))
        dEdu1 = np.atleast_2d(np.delete(dEdu1, -1))

        # Update output weights
        self.W_hidden_to_output -= self.learning_rate * (dEdu2.T.dot(self.o_hidden)).T

        # Update input weights
        self.W_input_to_hidden -= self.learning_rate * (dEdu1.T.dot(self.a_input)).T

        # Calculate error
        return 0.5 * sum((self.output - targets)**2)


    def train(self, data, validation_data):
        start_time = time.time()
        errors = []
        training_accuracies= []

        for it in range(self.iterations):
            np.random.shuffle(data)
            inputs = [ entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]

            error = 0.0
            for i in range(len(inputs)):
                input_ = inputs[i]
                target_ = targets[i]
                self.feed_forward(input_)
                error += self.back_propagate(target_)
            training_accuracies.append(self.predict(data))

            error = error / len(data)
            errors.append(error)

            # print("Iteration: %2d/%2d[==============] -Error: %5.10f -Training_Accuracy:  %s  -time: %2.2f " %(it+1,self.iterations, error, (self.predict(data)/len(data))*100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation

        plot_curve(np.arange(self.iterations) + 1, errors, "Error")
        plot_curve(np.arange(self.iterations) + 1, training_accuracies, "Training_Accuracy")


    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax(self.feed_forward(testcase[0]))
            count = count + 1 if (answer - prediction) == 0 else count
            count= count
        return count


    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'wi': self.W_input_to_hidden, 'wo': self.W_hidden_to_output}, f)


    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W_input_to_hidden = data['wi']
        self.W_hidden_to_output = data['wo']
