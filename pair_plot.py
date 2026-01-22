import sys
import os
import pandas as pd
from utils import data_tools, stats_tools
import matplotlib.pyplot as plt
import seaborn as sns

INCLUDE_FEATURES = [
	'Herbology', 'Defense Against the Dark Arts',
	'Divination', 'Muggle Studies', 'Ancient Runes',
	'Transfiguration']

# Utilisation de seaborn pour modelisation de pair plot a partir d'un df
def pair_plot(df):
	df = df[INCLUDE_FEATURES + ['Hogwarts House']]
	# df.dropna(inplace=True)
	sns.pairplot(df, hue='Hogwarts House', palette={
		'Gryffindor': 'red',
		'Slytherin': 'green',
		'Ravenclaw': 'blue',
		'Hufflepuff': 'gold'
	})

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
