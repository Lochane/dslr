from abc import ABC, abstractmethod

class BasePredictor(ABC):
	def __init__(self):
		raise NotImplementedError("The __init__ method must be implemented by subclasses.")

	@abstractmethod
	def predict(self):
		raise NotImplementedError("The predict method must be implemented by subclasses.")
