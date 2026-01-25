import numpy as np

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
		for _ in range(iterations): ## Gradient descendant
			print("\033[93mLoading iteration...\033[0m", end="\r")
			for i in range(n):
				prediction = self.forward(x[i])
				error = prediction - y[i]
				self.bias -= (learning_rate * error) / n ## Mise à jour du biais
				for j in range(len(self.weights)): ## Mise à jour des poids
					self.weights[j] -= (learning_rate * error * x[i][j]) / n
