import numpy as np
import sys
import pandas as pd
import json
from logisticNeuron import LogisticNeuron
from standardScaler import StandardScaler
from utils.stats_tools import ft_mean, ft_std_dev
from sklearn.metrics import accuracy_score
from logreg_predict import predict_house

INCLUDE_FEATURES = [
    'Herbology', 'Defense Against the Dark Arts',
    'Divination', 'Muggle Studies', 'Ancient Runes',
    'Transfiguration'
]

def main():

    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <path_to_csv>")
        sys.exit(1)

    try:
        df = pd.read_csv(sys.argv[1], index_col="Index")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    df = df.dropna(subset=['Hogwarts House'])
    split = int(len(df) * 0.2)
    train_df = df.iloc[split:]
    valid_df = df.iloc[:split]

    all_models = {}

    for house in train_df['Hogwarts House'].unique():
        scaler = StandardScaler(INCLUDE_FEATURES)
        x_train = scaler.fit_transform(train_df[INCLUDE_FEATURES].values.tolist())
        y_train = [1 if h == house else 0 for h in train_df['Hogwarts House']]
        neuron = LogisticNeuron([0] * len(INCLUDE_FEATURES), 0)
        neuron.fit(x_train, y_train, learning_rate=0.1, iterations=1000)
        all_models[house] = {
            "theta0": neuron.bias,
            "theta": neuron.weights,
            "x_mean": scaler.means,
            "x_std": scaler.stds
        }
        print(f"\033[92mModel trained for {house}\033[0m")
    
    results = []

    for _, row in valid_df.iterrows():
        student = row.to_dict()
        predicted_house = predict_house(student, all_models)
        results.append(predicted_house)

    accuracy = accuracy_score(valid_df['Hogwarts House'], results)
    print(f"\033[94mOverall validation accuracy: {accuracy * 100:.2f}%\033[0m")
    print("\033[92mSaving all models to all_thetas.json\033[0m")
    with open("all_thetas.json", "w") as f:
        json.dump(all_models, f, indent=2)



if __name__ == "__main__":
    main()