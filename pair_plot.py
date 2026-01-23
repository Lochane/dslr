import sys
import os
import pandas as pd
from utils import data_tools, stats_tools
import matplotlib.pyplot as plt

INCLUDE_FEATURES = [
	'Herbology', 'Defense Against the Dark Arts',
	'Divination', 'Muggle Studies', 'Ancient Runes',
	'Transfiguration']

# Utilisation de seaborn pour modelisation de pair plot a partir d'un df
def pair_plot(df):
	fig, axes = plt.subplots(len(INCLUDE_FEATURES), len(INCLUDE_FEATURES), figsize=(15, 15))
	for i, f1 in enumerate(INCLUDE_FEATURES):
		for j, f2 in enumerate(INCLUDE_FEATURES):
			ax = axes[i, j]
			if i == j:
				# Histogrammes sur la diagonale (même feature)
				gry = df[df["Hogwarts House"] == "Gryffindor"][f1].dropna()
				rav = df[df["Hogwarts House"] == "Ravenclaw"][f1].dropna()
				sly = df[df["Hogwarts House"] == "Slytherin"][f1].dropna()
				huf = df[df["Hogwarts House"] == "Hufflepuff"][f1].dropna()
				
				ax.hist(gry, bins=30, alpha=0.5, label='Gryffindor', color='red')
				ax.hist(rav, bins=30, alpha=0.5, label='Ravenclaw', color='blue')
				ax.hist(sly, bins=30, alpha=0.5, label='Slytherin', color='green')
				ax.hist(huf, bins=30, alpha=0.5, label='Hufflepuff', color='yellow')
				ax.set_ylabel(f1 if j == 0 else '')
				ax.set_xlabel(f1 if i == len(INCLUDE_FEATURES) - 1 else '')
			else:
				# Scatter plots pour les autres cases
				for house, group in df.groupby('Hogwarts House'):
					colors = {'Gryffindor': 'red', 'Ravenclaw': 'blue', 'Slytherin': 'green', 'Hufflepuff': 'yellow'}
					ax.scatter(group[f2], group[f1], label=house, alpha=0.5, color=colors.get(house, 'gray'), s=10)
				ax.set_ylabel(f1 if j == 0 else '')
				ax.set_xlabel(f2 if i == len(INCLUDE_FEATURES) - 1 else '')
				

	plt.tight_layout()
	outdir = os.path.join("plots")
	os.makedirs(outdir, exist_ok=True)
	plt.savefig(os.path.join(outdir, "pair_plot.png"))
	plt.close()

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python pair_plot.py <path_to_csv>")
		sys.exit(1)

	df = pd.read_csv(sys.argv[1], index_col="Index")
	pair_plot(df)
