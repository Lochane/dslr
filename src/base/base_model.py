from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
	"""BaseModel is the base class for all models in the project. It provides common functionality and serves as a parent class for specific model implementations.
	"""
	def __init__(self):
		"""Initializes the BaseModel instance. This method can be overridden by subclasses to perform specific initialization tasks.
		"""
		raise NotImplementedError("The __init__ method must be implemented by subclasses.")

	@abstractmethod
	def predict(self, x: np.array):
		"""Generates predictions based on the input data. This method should be implemented by subclasses to define the prediction process.

		Args:
			input_data: The input data for which predictions are to be generated.

		Returns:
			The predicted output based on the input data.
		"""
		raise NotImplementedError("The predict method must be implemented by subclasses.")