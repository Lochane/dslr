import math
import numpy as np
from src.base import BasePreprocessor

class StandardScaler(BasePreprocessor):
	"""Standard Scaler class for normalizing data.""" 
	def __init__(self, means: np.ndarray=None, stds: np.ndarray=None):
		"""Initialize the StandardScaler with mean and std. If mean and std are not provided, they will be computed from the data during fitting.
			Args:
				mean (np.ndarray): The mean of the features.
				std (np.ndarray): The standard deviation of the features.
		"""
		self.means = means
		self.stds = stds

	def fit(self, data: np.ndarray):
		"""Compute the mean and standard deviation of the data.
			Args:
				data (np.ndarray): The data to compute the mean and standard deviation from.
		"""
		self.means = np.nanmean(data, axis=0)
		self.stds = np.nanstd(data, axis=0)

	def transform(self, data: np.ndarray) -> np.ndarray:
		"""Transform the data using the computed mean and standard deviation.
			Args:
				data (np.ndarray): The data to transform.
			Returns:
				np.ndarray: The transformed data.
		"""
		# Ensure we work on a float copy to handle NaNs correctly
		x = data.astype(float, copy=True)

		# Replace NaNs with the corresponding feature mean
		nan_mask = np.isnan(x)
		x = np.where(nan_mask, self.means, x)

		# Avoid division by zero by using 1.0 where std is zero
		std_safe = np.where(self.stds != 0, self.stds, 1.0)
		return (x - self.means) / std_safe

	def fit_transform(self, data: np.ndarray) -> np.ndarray:
		"""Fit the scaler to the data and transform it.
			Args:
				data (np.ndarray): The data to fit and transform.
			Returns:
				np.ndarray: The transformed data.
		"""
		self.fit(data)
		return self.transform(data)