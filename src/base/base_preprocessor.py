from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
	"""Base class for preprocessors. Preprocessors are used to preprocess the data before it is fed to the model.
	"""

	def __init__(self):
		"""Initializes the BasePreprocessor instance. This method can be overridden by subclasses to perform specific initialization tasks.
		"""
		pass

	@abstractmethod
	def fit(self, data):
		"""Fits the preprocessor to the input data. This method should be implemented by subclasses to define the fitting steps.

		Args:
			data: The input data to fit the preprocessor on.
		"""
		raise NotImplementedError("The fit method must be implemented by subclasses.")

	@abstractmethod
	def transform(self, data):
		"""Transforms the input data. This method should be implemented by subclasses to define the transformation process.

		Args:
			data: The input data to be transformed.

		Returns:
			The transformed data based on the input data.
		"""
		raise NotImplementedError("The transform method must be implemented by subclasses.")

	def fit_transform(self, data):
		"""Fits the preprocessor to the input data and then transforms it. This method can be implemented by subclasses to define the combined fitting and transformation process.

		Args:
			data: The input data to fit and transform.
		"""
		self.fit(data)
		return self.transform(data)