import numpy as np
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
			self.means.append(ft_mean(feature_value))
			self.stds.append(ft_std_dev(feature_value))
	
	def transform(self, data):
		self.normValues = []
		for student in range(len(data)):
			student_features = []
			for i, feature in enumerate(self.features):
				val = data[student][i]
				if val is None or (isinstance(val, float) and val != val):
					val = self.means[i]
				std = self.stds[i] if self.stds[i] != 0 else 1
				student_features.append((val - self.means[i]) / std)
			self.normValues.append(student_features)
		return self.normValues

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)
