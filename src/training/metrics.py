import numpy as np

def binary_cross_entropy_loss(y_true, y_pred):
	y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
	return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Mean Squared Error function.
		Args:
			y_true (np.ndarray): The true target values.
			y_pred (np.ndarray): The predicted target values.
		Returns:
			float: The mean squared error between the true and predicted values.
	"""
	return np.mean((y_true - y_pred) ** 2)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Root Mean Squared Error function.
		Args:
			y_true (np.ndarray): The true target values.
			y_pred (np.ndarray): The predicted target values.
		Returns:
			float: The root mean squared error between the true and predicted values.
	"""
	return np.sqrt(mse(y_true, y_pred))
