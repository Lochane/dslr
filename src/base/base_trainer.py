from abc import ABC, abstractmethod
from src.base import BaseModel

class BaseTrainer(ABC):
	"""Base class for training machine learning models."""
	def __init__(self, model: BaseModel, learning_rate: float = 0.01, iterations: int = 1000):
		"""
		Initializes the trainer with a given model.
		
		Args:
			model: The machine learning model to be trained.
			learning_rate: The learning rate for gradient descent.
			iterations: The number of iterations to perform during training.
		"""
		self.model = model
		self.learning_rate = learning_rate
		self.iterations = iterations
	
	@abstractmethod
	def train(self):
		"""
		Trains the model using the provided training data.
		"""
		raise NotImplementedError("The train method must be implemented by subclasses.")