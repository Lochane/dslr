import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.stats_tools import ft_mean, ft_std_dev

class StandardScaler:
	def __init__(self, features):
		self.features = features
		self.means = []
		self.stds = []
		self.normValues = []

	def fit(self, data):
		for i in range(len(self.features)):
			feature_value = [data[j][i] for j in range(len(data))]
			self.means.append(ft_mean(feature_value)) ## Calculer la moyenne
			self.stds.append(ft_std_dev(feature_value)) ## Calculer l'écart-type
	
	## Normalisation des données
	def transform(self, data):
		self.normValues = []
		for student in range(len(data)):
			student_features = []
			for i, feature in enumerate(self.features):
				val = data[student][i]
				if val is None or (isinstance(val, float) and val != val): ## gérer les NaN
					val = self.means[i] ## remplacer par la moyenne
				std = self.stds[i] if self.stds[i] != 0 else 1 ## éviter la division par zéro
				student_features.append((val - self.means[i]) / std) ## normalisation
			self.normValues.append(student_features)
		return self.normValues

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)
