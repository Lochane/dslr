import numpy as np
from src.base import BaseModel, BaseTrainer, BasePreprocessor
from src.persistence import save_model
from src.training.metrics import binary_cross_entropy_loss


class GradientDescentTrainer(BaseTrainer):
	"""Trainer for models using gradient descent."""
	
	def __init__(self, model: BaseModel, learning_rate: float = 0.01, iterations: int = 1000):
		"""Init GradientDescentTrainer 
			Args:
				model: The machine learning model to be trained.
				learning_rate: The learning rate for gradient descent.
				iterations: The number of iterations to perform during training.
		"""
		super().__init__(model, learning_rate, iterations)
	
	def train(self, x: np.ndarray, y: np.ndarray) -> None:
		"""Train the model using gradient descent.
			Args:
				x (np.ndarray): The input features for training.
				y (np.ndarray): The target values for training.
		"""
		n = len(x)
		for ite in range(self.iterations):
			y_pred = self.model.predict(x)
			error = y_pred - y
			self.model.bias -= self.learning_rate * np.mean(error)
			self.model.weights -= self.learning_rate * np.dot(x.T, error) / n
			if ite % 100 == 0:
				y_pred = self.model.predict(x)
				print(f"Loss: {binary_cross_entropy_loss(y, y_pred):.6f} | LR: {self.learning_rate} | Iter: {ite:>4}/{self.iterations}")
			
		y_pred = self.model.predict(x)
		print(f"\033[92mFinal Loss: {binary_cross_entropy_loss(y, y_pred):.6f}\033[0m")

class OneVsAllTrainer(BaseTrainer):
	def __init__(self, trainer: BaseTrainer, scaler: BasePreprocessor ,learning_rate: float = 0.01, iterations: int = 1000):
		super().__init__(trainer.model, learning_rate, iterations)
		self.trainer = trainer
		self.scaler = scaler

	def train(self, x: np.ndarray, y: np.ndarray) -> list:
		all_models = {}
		json_conf = {
			"scaler": {
				"x_mean": self.scaler.means.tolist(),
				"x_std": self.scaler.stds.tolist()
			}
		}

		for yi in np.unique(y):
			print(f"\033[93mTraining model for {yi}\033[0m")
			y_binary = (y == yi).astype(int)
			self.trainer.model.weights = np.zeros(x.shape[1])
			self.trainer.model.bias = 0.0
			self.trainer.train(x, y_binary)
			all_models[yi] = {
				"theta0": self.trainer.model.bias.tolist(),
				"theta": self.trainer.model.weights.tolist(),
			}

		json_conf["models"] = all_models
		save_model(json_conf, "data/models/all_thetas.json")
		return all_models

