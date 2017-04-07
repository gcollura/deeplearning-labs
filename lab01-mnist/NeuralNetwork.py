from __future__ import print_function
import time
import random
import numpy as np
from utils import *
from transfer_functions import *


class NeuralNetwork(object):

    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size,
            iterations = 50, learning_rate = 0.1,
            tfunction = sigmoid, dtfunction = dsigmoid):
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
        self.tfunction = tfunction
        self.dtfunction = dtfunction

        # initialize arrays
        self.input = input_layer_size + 1   # +1 for the bias node in the input layer
        self.hidden = hidden_layer_size + 1 # +1 for the bias node in the hidden layer
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden = np.ones(self.hidden)
        self.a_output = np.ones(self.output)

        # create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input**(1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden - 1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)


    def weights_initialisation(self, wi, wo):
        self.W_input_to_hidden = wi # weights between input and hidden layers
        self.W_hidden_to_output = wo # weights between hidden and output layers


    def feed_forward(self, inputs):
        # Compute input activations
        self.a_input = np.array(inputs)
        if self.a_input.size < self.input:
            self.a_input = np.append(self.a_input, 1)
        self.a_input = np.atleast_2d(self.a_input)

        # Compute hidden activations
        self.a_hidden = self.a_input.dot(self.W_input_to_hidden)
        self.o_hidden = self.tfunction(self.a_hidden)
        if len(self.o_hidden) < self.hidden:
            self.o_hidden = np.append(self.o_hidden, 1)
        self.o_hidden = np.atleast_2d(self.o_hidden)

        # Compute output activations
        self.a_output = self.o_hidden.dot(self.W_hidden_to_output)
        self.output = self.tfunction(self.a_output)
        return self.output


    def back_propagate(self, targets):
        # Calculate error terms for output
        dEdu2 = np.multiply(self.output - targets, self.dtfunction(self.output))
        dEdu2 = np.atleast_2d(dEdu2)

        # Calculate error terms for hidden
        dEdu1 = np.multiply(dEdu2.dot(self.W_hidden_to_output.T), self.dtfunction(self.o_hidden))
        dEdu1 = np.atleast_2d(np.delete(dEdu1, -1))

        # Update output weights
        self.W_hidden_to_output -= self.learning_rate * (dEdu2.T.dot(self.o_hidden)).T

        # Update input weights
        self.W_input_to_hidden -= self.learning_rate * (dEdu1.T.dot(self.a_input)).T

        # Calculate error
        return 0.5 * np.sum((self.output - targets)**2)


    def print_iteration(self, step, iterations, error, t_accuracy, v_accuracy, time):
        print('[%2d/%2d]' % (step, iterations), end=' ')
        print('Error: %5.5f\tTraining Acc.: %2.2f\tValidation Acc.: %2.2f\tTime: %3.2f' % (error, t_accuracy, v_accuracy, time))


    def train(self, data, validation_data, print_step = 1):
        start_time = time.time()
        errors = []
        training_accuracies= []
        validation_accuracies= []

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
            validation_accuracies.append(self.predict(validation_data))
            error = error / len(data)
            errors.append(error)

            if it % print_step == 0 or it == self.iterations - 1:
                self.print_iteration(it+1, self.iterations, error,
                        (self.predict(data)/len(data))*100,
                        (self.predict(validation_data)/len(validation_data))*100,
                        time.time() - start_time)
            # you can add test_accuracy and validation accuracy for visualisation

        fig = plt.figure(figsize=(20, 8))
        plt.subplot(1, 3, 1)
        plot_curve(np.arange(self.iterations) + 1, errors, "Error", fig)
        plt.subplot(1, 3, 2)
        plot_curve(np.arange(self.iterations) + 1, training_accuracies, "Training_Accuracy", fig)
        plt.subplot(1, 3, 3)
        plot_curve(np.arange(self.iterations) + 1, validation_accuracies, "Validation_Accuracy", fig)
        plt.show()

        return validation_accuracies


    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax(testcase[1])
            prediction = np.argmax(self.feed_forward(testcase[0]))
            count = count + 1 if (answer - prediction) == 0 else count
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


class ModNeuralNetwork(NeuralNetwork):

    def __init__(self, input_layer_size, first_hidden_layer_size,
            second_hidden_layer_size, output_layer_size, iterations = 50,
            learning_rate = 0.1, tfunction = sigmoid, dtfunction = dsigmoid):
        """
        input: number of input neurons
        first_hidden: number of first layer of hidden neurons
        second_hidden: number of second layer of hidden neurons
        output: number of output neurons
        iterations: how many iterations
        learning_rate: initial learning rate
        tfunction: transfer function
        """
        # initialize parameters
        self.iterations = iterations        # iterations
        self.learning_rate = learning_rate
        self.tfunction = tfunction
        self.dtfunction = dtfunction

        # initialize arrays
        self.input = input_layer_size + 1   # +1 for the bias node in the input layer
        self.first_hidden = first_hidden_layer_size + 1 # +1 for the bias node in the hidden layer
        self.second_hidden = second_hidden_layer_size + 1
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_first_hidden = np.ones(self.first_hidden)
        self.a_second_hidden = np.ones(self.second_hidden)
        self.a_output = np.ones(self.output)

        #create randomized weights Yann Lecun method in 1988's paper (Default values)
        input_range = 1.0 / self.input**(1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale =
                input_range, size = (self.input, self.first_hidden - 1))
        self.W_hidden_to_hidden = np.random.uniform(size = (self.first_hidden,
            self.second_hidden - 1)) / np.sqrt(self.first_hidden)
        self.W_hidden_to_output = np.random.uniform(size = (self.second_hidden,
            self.output)) / np.sqrt(self.second_hidden)


    def weights_initialisation(self, wi, wh, wo):
        self.W_input_to_hidden = wi # weights between input and hidden layers
        self.W_hidden_to_hidden = wh # weights between the two hidden layers
        self.W_hidden_to_output = wo # weights between hidden and output layers


    def feed_forward(self, inputs):
        # Compute input activations
        self.a_input = np.array(inputs)
        if self.a_input.size < self.input:
            self.a_input = np.append(self.a_input, 1)
        self.a_input = np.atleast_2d(self.a_input)

        # Compute first hidden activations
        self.a_first_hidden = self.a_input.dot(self.W_input_to_hidden)
        self.o_first_hidden = self.tfunction(self.a_first_hidden)
        if len(self.o_first_hidden) < self.first_hidden:
            self.o_first_hidden = np.append(self.o_first_hidden, 1)
        self.o_first_hidden = np.atleast_2d(self.o_first_hidden)

        # Compute second hidden activations
        self.a_second_hidden = self.o_first_hidden.dot(self.W_hidden_to_hidden)
        self.o_second_hidden = self.tfunction(self.a_second_hidden)
        if len(self.o_second_hidden) < self.second_hidden:
            self.o_second_hidden = np.append(self.o_second_hidden, 1)
        self.o_second_hidden = np.atleast_2d(self.o_second_hidden)

        # Compute output activations
        self.a_output = self.o_second_hidden.dot(self.W_hidden_to_output)
        self.output = self.tfunction(self.a_output)
        return self.output


    def back_propagate(self, targets):
        # Calculate error terms for output
        dEdu3 = np.multiply(self.output - targets, self.dtfunction(self.output))
        dEdu3 = np.atleast_2d(dEdu3)

        # Calculate error terms for hidden
        dEdu2 = np.multiply(dEdu3.dot(self.W_hidden_to_output.T), self.dtfunction(self.o_second_hidden))
        dEdu2 = np.atleast_2d(np.delete(dEdu2, -1))

        dEdu1 = np.multiply(dEdu2.dot(self.W_hidden_to_hidden.T), self.dtfunction(self.o_first_hidden))
        dEdu1 = np.atleast_2d(np.delete(dEdu1, -1))

        # Update output weights
        self.W_hidden_to_output -= self.learning_rate * (dEdu3.T.dot(self.o_second_hidden)).T

        # Update hidden_weights
        self.W_hidden_to_hidden -= self.learning_rate * (dEdu2.T.dot(self.o_first_hidden)).T

        # Update input weights
        self.W_input_to_hidden -= self.learning_rate * (dEdu1.T.dot(self.a_input)).T

        # Calculate error
        return 0.5 * np.sum((self.output - targets)**2)


    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'wi': self.W_input_to_hidden, 'wh':
                self.W_hidden_to_hidden, 'wo': self.W_hidden_to_output}, f)


    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W_input_to_hidden = data['wi']
        self.W_hidden_to_hidden = data['wh']
        self.W_hidden_to_output = data['wo']
