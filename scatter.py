import sys
import itertools
from utils import data_tools, stats_tools
import matplotlib.pyplot as plt
import pandas as pd
import os

# Formatage des donnees
def pd_prep_csv(path):
    dataset = pd.read_csv(path, index_col="Index")
    print(f"Shape avant dropna {dataset.shape}")
    dataset.dropna(inplace=True)
    print(f"Shape after dropna {dataset.shape}")
    return dataset


def prep_scatter(df):
    df.drop(['First Name' ,'Last Name', 'Best Hand', 'Birthday'], axis=1, inplace=True)
    print("Colums head after drop", df.keys())
    

# Choix des deux features avec le plus de correlation

def get_corr_feature(df):
    features = df.columns.tolist()
    features = features[1:]
    best_pair = None
    best_corr = 0

    for f1, f2 in itertools.combinations(features, 2):
        x = df[f1].tolist()
        y = df[f2].tolist()
        r = stats_tools.ft_correlation(x, y)
        if abs(r) > best_corr:
            best_corr = abs(r)
            best_pair = (f1, f2)


    print("Les deux features les plus similaires :", best_pair)
    print("Corrélation absolue :", best_corr)
    return best_pair


# Plotting
def plt_scatter(pair, df):
    colors = {"Gryffindor":"red", "Slytherin":"green", "Ravenclaw":"blue", "Hufflepuff":"gold"}
    feat1, feat2 = pair
    subset = df[["Hogwarts House", feat1, feat2]]
    for house in df["Hogwarts House"].unique():
        tmp = subset[subset["Hogwarts House"] == house]
        plt.scatter(tmp[feat1], tmp[feat2], color=colors[house], label=house, alpha=0.5, edgecolors='none')

    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.legend()
    # plt.show()
    plt.tight_layout()
    outdir = os.path.join("plots")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "scatter_plot.png"))
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <filename>")
        sys.exit(1)

    df = pd_prep_csv(sys.argv[1])
    print("Clés :", df.keys())
    print("Exemple d'une colonne :", list(df.keys())[0])
    print("Dataframe de pandas")
    print(df.head)
    print(df.columns)
    prep_scatter(df)
    corr_feature = get_corr_feature(df)
    plt_scatter(corr_feature, df)
