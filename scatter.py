import sys
from utils import data_tools, stats_tools
from histogram import prepvalues
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <filename>")
        sys.exit(1)

    dataset = data_tools.load_csv(sys.argv[1])
    df = pd.read_csv(sys.argv[1])
    print(f'type de dataset {type(dataset)}')
    prep_dataset = prepvalues(dataset)
    print(f'type du dataset traiter {type(prep_dataset)}')
    print("Clés :", dataset.keys())
    print("Exemple d'une colonne :", list(dataset.keys())[0])
    print("Premières valeurs de cette colonne :", dataset[list(prep_dataset.keys())[0]][:5])
    print("Dataframe de pandas")
    print(df.head)
    print(df.columns)


