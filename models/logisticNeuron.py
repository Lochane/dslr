import numpy as np

def binary_cross_entropy_loss(y_true, y_pred):
	y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
	return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class Neuron:
	def __init__(self, weights, bias):
		self.weights = weights
		self.bias = bias

	def forward(self, inputs):
		raise NotImplementedError("Forward method not implemented.")

class LogisticNeuron(Neuron):
	
	def forward(self, inputs):
		score = self.bias
		for i in range(len(self.weights)):
			score += self.weights[i] * inputs[i] ## Somme des poids * entrées
		return 1 / (1 + np.exp(-score)) ## Activation sigmoïde
			

	def fit(self, x, y, learning_rate=0.01, iterations=1000):
		n = len(x)
		for ite in range(iterations): ## Gradient descendant
			print("\033[93mTraining Model...\033[0m", end="\r")
			sum_error_b = 0
			sum_error_w = [0.0] * len(self.weights)
			for i in range(n):
				prediction = self.forward(x[i])
				error = prediction - y[i]
				sum_error_b += error
				for j in range(len(self.weights)):
					sum_error_w[j] += error * x[i][j]

			self.bias -= learning_rate * (sum_error_b / n) ## Mise à jour du biais
			for j in range(len(self.weights)): ## Mise à jour des poids
				self.weights[j] -= learning_rate * (sum_error_w[j] / n)
