import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

INCLUDE_FEATURES = [
	'Herbology', 'Defense Against the Dark Arts',
	'Divination', 'Muggle Studies', 'Ancient Runes',
	'Transfiguration'
]


def sigmoide(score):
	return 1 / (1 + np.exp(-score))


def load_all_models(filename="all_thetas.json"):
	with open(filename, "r") as f:
		models = json.load(f)
	return models


# application de la formule de prediction sur une maison
def predict_proba(student, theta0, theta, x_mean, x_std):
	score = theta0
	for i, feature in enumerate(INCLUDE_FEATURES):
		val = student.get(feature)
		if val is None or (isinstance(val, float) and np.isnan(val)):  ## on remplace les donnee manquante par la moyenne du reste de la table
			val = x_mean[i]
		std = x_std[i] if x_std[i] != 0 else 1
		val_norm = (val - x_mean[i]) / std
		score += theta[i] * val_norm
	return sigmoide(score)


# Recherche de la maison avec la probabilite la plus grande
def predict_house(student, models):
	probs = {}
	for house, model in models.items():
		theta0 = model["theta0"]
		theta = model["theta"]
		x_mean = model["x_mean"]
		x_std = model["x_std"]
		probs[house] = predict_proba(student, theta0, theta, x_mean, x_std)
	return max(probs, key=probs.get)


def main():
	if len(sys.argv) != 2:
		print("Usage: python logreg_predict.py <path_to_csv>")
		sys.exit(1)

	try:
		df = pd.read_csv(sys.argv[1], index_col="Index")
	except Exception as e:
		print(f"Error reading the CSV file: {e}")
		sys.exit(1)

	models = load_all_models()
	results = []

	print("\033[93mPredicting houses...\033[0m")
	for idx, row in df.iterrows():
		student = row.to_dict()
		house = predict_house(student, models)
		results.append(house)

	df["Hogwarts House Predicted"] = results
	df[["Hogwarts House Predicted"]].to_csv("houses.csv")

	print("\033[92mPredictions saved to houses.csv\033[0m")
	# Vérifier si la colonne réelle existe et contient des valeurs
	if "Hogwarts House" in df.columns and df["Hogwarts House"].dropna().any():
		accuracy = accuracy_score(df["Hogwarts House"], df["Hogwarts House Predicted"])
		print(f"Accuracy: {accuracy:.4f}")
	else:
		print("No ground truth available to compute accuracy.")


if __name__ == "__main__":
	main()

