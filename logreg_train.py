import numpy as np
import sys
import pandas as pd
import json
from utils.stats_tools import ft_mean, ft_std_dev

INCLUDE_FEATURES = [
	'Herbology', 'Defense Against the Dark Arts',
	'Divination', 'Muggle Studies', 'Ancient Runes',
	'Transfiguration']

def sigmoide(score):
	return 1 / (1 + np.exp(-score))


def ft_logistic_regression(x, y, learning_rate=0.01, iterations=1000):

	x_mean = []
	x_std = []
	n_features = len(INCLUDE_FEATURES)
	
	for feature in range(n_features):
		feature_values = [x[i][feature] for i in range(len(x))]
		x_mean.append(ft_mean(feature_values))
		x_std.append(ft_std_dev(feature_values))

	x_norm = []
	for student in range(len(x)):
		student_features = []
		for feature in range(n_features):
			val = x[student][feature]
			if val is None or (isinstance(val, float) and val != val):
				val = x_mean[feature]
			std = x_std[feature] if x_std[feature] != 0 else 1
			student_features.append((val - x_mean[feature]) / std)
		x_norm.append(student_features)

	theta0 = 0
	theta = [0] * n_features
	n = len(x_norm)

	for _ in range(iterations):
		for student in range(n):
			score = theta0
			for feature in range(n_features):
				score += theta[feature] * x_norm[student][feature]
			prediction = sigmoide(score)
			error = prediction - y[student]
			theta0 -= (learning_rate * error) / n
			for feature in range(n_features):
				theta[feature] -= (learning_rate * error * x_norm[student][feature]) / n

	return theta0, theta

if __name__ == "__main__":
		# try :
		if len(sys.argv) != 2:
			print("Usage: python logreg_train.py <path_to_csv>")
			sys.exit(1)
		df = pd.read_csv(sys.argv[1], index_col="Index")
			# except FileNotFoundError:
			# 	print(f"File {sys.argv[1]} not found.")
			# 	sys.exit(1)
			# except pd.errors.ParserError:
			# 	print("Error: File is not a valid CSV format")
			# 	sys.exit(1)
			# except Exception as e:
			# 	print(f"Unexpected error: {e}")
			# 	sys.exit(1)

		df = df.dropna(subset=['Hogwarts House'])


		for house in df['Hogwarts House'].unique():
			x = df[INCLUDE_FEATURES].values.tolist()
			y = [1 if h == house else 0 for h in df['Hogwarts House']]
			theta0, theta = ft_logistic_regression(x, y)
			params = {"theta0": theta0, "theta": theta}
			with open("thetas_" + house.lower() + ".json", "w") as f:
				json.dump(params, f)

