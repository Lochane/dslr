import numpy as np
from src.base import BasePredictor, BaseModel

class OneVsAllPredictor(BasePredictor):
	def __init__(self, models: dict[str, BaseModel]):
		self.models = models

	def predict(self, x: np.ndarray):
		probs = np.zeros((len(self.models), len(x)))
		for index, model in enumerate(self.models.values()):
			probs[index] = model.predict(x)
		keys = list(self.models.keys())
		probs = probs.T
		probs = np.argmax(probs, axis=1)
		return np.array(keys)[probs]