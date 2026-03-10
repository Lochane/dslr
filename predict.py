import sys
import numpy as np
import pandas as pd
from src.models import LogisticRegression
from src.preprocessing import StandardScaler, load_csv
from src.persistence import load_model
from src.predict import OneVsAllPredictor



INCLUDE_FEATURES = [
	'Herbology', 'Defense Against the Dark Arts',
	'Divination', 'Muggle Studies', 'Ancient Runes',
	'Transfiguration'
]

def main():
	if len(sys.argv) != 2:
		print("Usage: python predict.py <path_to_csv>")
		sys.exit(1)

	try:
		df = load_csv(sys.argv[1])
		all_params = load_model("data/models/all_thetas.json")
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)

	scaler_params = all_params["scaler"]
	scaler = StandardScaler(np.asarray(scaler_params["x_mean"]), np.asarray(scaler_params["x_std"]))

	models_params = all_params["models"]
	models = {}
	for key, params in models_params.items():
		models[key] = LogisticRegression(weights=np.asarray(params["theta"]), bias=np.asarray(params["theta0"]))

	raw_students = df[INCLUDE_FEATURES].to_numpy()
	norm_students = scaler.transform(raw_students)
	predictor = OneVsAllPredictor(models)
	results  = predictor.predict(norm_students)
	
	df["Hogwarts House Predicted"] = results
	df[["Hogwarts House Predicted"]].to_csv("houses.csv")

	print("\033[92mPredictions saved to houses.csv\033[0m")

if __name__ == "__main__":
	main()
