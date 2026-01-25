import sys
import os
from utils import data_tools, stats_tools
import pandas as pd
import matplotlib.pyplot as plt
import os

def histogram(dataset):
    topics = [c for c in data_tools.to_numeric_list(dataset) if c.lower() != "index"]
    n = len(topics)

    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # séparation des données par maison
    for i, topic in enumerate(topics):
        gry = [v for j, v in enumerate(dataset[topic]) if dataset["Hogwarts House"][j] == "Gryffindor" and v is not None]
        rav = [v for j, v in enumerate(dataset[topic]) if dataset["Hogwarts House"][j] == "Ravenclaw" and v is not None]
        sly = [v for j, v in enumerate(dataset[topic]) if dataset["Hogwarts House"][j] == "Slytherin" and v is not None]
        huf = [v for j, v in enumerate(dataset[topic]) if dataset["Hogwarts House"][j] == "Hufflepuff" and v is not None]
        if not (gry or rav or sly or huf):
            continue

    # Tracer les histogrammes
        ax = axes[i]
        ax.hist(gry, bins=30, alpha=0.5, label='Gry', color='r')
        ax.hist(rav, bins=30, alpha=0.5, label='Rav', color='b')
        ax.hist(sly, bins=30, alpha=0.5, label='Sly', color='g')
        ax.hist(huf, bins=30, alpha=0.5, label='Huf', color='y')
        ax.set_title(topic)
        ax.legend(fontsize=8)

    # Supprimer les axes inutilisés
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    outdir = os.path.join("plots")
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "all_histograms.png"))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <filename>")
        sys.exit(1)

    try:
        df = pd.read_csv(sys.argv[1], index_col="Index")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    histogram(df)