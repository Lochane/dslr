import numpy as np
from base.src import BaseModel

class LogisticRegression(BaseModel):
	def __init__(self, weights: np.ndarray, bias: float):
		self.weights = weights
		self.bias = bias

	def _sigmoid(self, score :np.ndarray) -> np.ndarray:
		return 1/ (1 + np.exp(-score))

	def predict(self, features: np.ndarray) -> np.ndarray:
		score = self.bias + np.dot(features, self.weights)
		return self._sigmoid(score)

