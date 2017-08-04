import math
import numpy as np
from .functions import radial_based_gaussian_function


class RBF():
	ciclos = 1000
	num_training = 7	# Atribui o número de exemplos de treinamento.

    def __init__(self, input_number, output_number, hidden_neurons_number):
        self.inputNumber = input_number
        self.outputNumber = output_number
        self.hiddenNeuronsNumber = hidden_neurons_number
        self.input_values = []
        self.centers = []
        self.radius = []
        self.weights = []

     def initialize(self):
     	self.input_values = np.mgrid[-1:1:complex(0, n)].reshape(n, 1)
     	self.centers = np.random.randn(self.hidden_neurons_number)
     	self.weights = np.random.randn(self.hidden_neurons_number)

     def get_output(self, x):
     	return math.sin(3 * (x + 0.5)**3 - 1)

    def radial_based_activation_function(self, input_value, center, radius):
        return radial_based_gaussian_function(input_value, center, radius)

    def calculate_activation(self, input_values):
		# calculates activations of hidden layer of RBFs
		inputs_hidden_layer = []
		for index, input_value in enumerate(input_values):
			inputs_hidden_layer.append(
				radial_based_gaussian_function(
					input_value, self.centers[index], self.radius[index]
				)
			)
		return inputs_hidden_layer

	def calculate_output(self, inputs_hidden_layer):
		return np.dot(inputs_hidden_layer, self.weights)

	def predict(self):
		inputs_hidden_layer = calculate_activation(self.input_values)
		output = calculate_output(inputs_hidden_layer)
		net2 = sum(output)
		return (1 / (1 + math.exp(-net2)))

	def train(self):
		for i in range(self.ciclos):
			rnd_idx = random.permutation(self.input_values.shape[0])[:self.num_training]
			training_values = [self.input_values[i,:] for i in rnd_idx]
			
			net = calculate_activation(training_values)
			output = calculate_output(net)
			error = get_output(training_values) - output

			adjust_centers()
			adjust_radius()
			adjust_weights()

	def adjust_centers():
		self.centers = 