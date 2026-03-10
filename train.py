import sys
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from src.models import LogisticRegression
from src.preprocessing import StandardScaler, load_csv
from src.training import OneVsAllTrainer, GradientDescentTrainer, OneVsAllPredictor
from sklearn.metrics import accuracy_score


INCLUDE_FEATURES = [
	'Herbology', 'Defense Against the Dark Arts',
	'Divination', 'Muggle Studies', 'Ancient Runes',
	'Transfiguration'
]

def main():

	if len(sys.argv) != 2:
		print("Usage: python train.py <path_to_csv>")
		sys.exit(1)

	try:
		df = load_csv(sys.argv[1])
	except Exception as e:
		print(f"Error reading the CSV file: {e}")
		sys.exit(1)
	

	df = df.dropna(subset=['Hogwarts House'])
	split = int(len(df) * 0.2)
	train_df = df.iloc[split:]
	valid_df = df.iloc[:split]

	scaler = StandardScaler()
	raw_x = train_df[INCLUDE_FEATURES].to_numpy()
	x_train = scaler.fit_transform(raw_x)
	y_train = train_df['Hogwarts House'].to_numpy()
	trainer = OneVsAllTrainer(GradientDescentTrainer(LogisticRegression()), scaler)
	all_models = trainer.train(x_train, y_train, valid_df)
	
	models = {}
	for key, params in all_models.items():
		models[key] = LogisticRegression(weights=np.asarray(params["theta"]), bias=np.asarray(params["theta0"]))

	raw_students = valid_df[INCLUDE_FEATURES].to_numpy()
	norm_students = scaler.transform(raw_students)
	predictor = OneVsAllPredictor(models)
	results  = predictor.predict(norm_students)

	accuracy = accuracy_score(valid_df['Hogwarts House'], results)
	print(f"\033[94mOverall validation accuracy: {accuracy * 100:.2f}%\033[0m")
	print("\033[92mSaving all models to all_thetas.json\033[0m")

if __name__ == "__main__":
	main()