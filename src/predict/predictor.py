import numpy as np
from src.base import BasePredictor, BaseModel

class OneVsAllPredictor(BasePredictor):
	def __init__(self, models: dict[str, BaseModel]):
		self.models = models

	def predict(self, x: np.ndarray) -> list:
		results = []
		for xi in x:
			probs = {}
			for key, model in self.models.items():
				probs[key] = model.predict(xi)
			results.append(max(probs, key=probs.get))
		return results