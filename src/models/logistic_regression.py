import numpy as np
from src.base import BaseModel

class LogisticRegression(BaseModel):
	def __init__(self, weights: np.ndarray = None, bias: float = 0):
		self._weights = weights if weights is not None else np.array([])
		self._bias = bias

	@property
	def weights(self):
		return self._weights

	@weights.setter
	def weights(self, weights: np.ndarray):
		self._weights = weights

	@property
	def bias(self):
		return self._bias

	@bias.setter
	def bias(self, bias: np.ndarray):
		self._bias = bias

	def _sigmoid(self, score :np.ndarray) -> np.ndarray:
		return 1/ (1 + np.exp(-score))

	def predict(self, features: np.ndarray) -> np.ndarray:
		score = self._bias + np.dot(features, self._weights)
		return self._sigmoid(score)

