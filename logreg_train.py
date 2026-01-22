import numpy as np
import sys
import pandas as pd
import json
from utils.stats_tools import ft_mean, ft_std_dev

INCLUDE_FEATURES = [
    'Herbology', 'Defense Against the Dark Arts',
    'Divination', 'Muggle Studies', 'Ancient Runes',
    'Transfiguration'
]

def sigmoide(score):
    return 1 / (1 + np.exp(-score))


def ft_logistic_regression(x, y, learning_rate=0.01, iterations=1000):

    # Data gathering for models application : Mean, std, norm
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

    # Gradient descent
    theta0 = 0
    theta = [0] * n_features
    n = len(x_norm)

    for _ in range(iterations):
        print("\033[93mLoading iteration...\033[0m", end="\r")
        for student in range(n):
            score = theta0
            for feature in range(n_features):
                score += theta[feature] * x_norm[student][feature]
            prediction = sigmoide(score)
            error = prediction - y[student]
            theta0 -= (learning_rate * error) / n
            for feature in range(n_features):
                theta[feature] -= (learning_rate * error * x_norm[student][feature]) / n

    return theta0, theta, x_mean, x_std


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <path_to_csv>")
        sys.exit(1)

    try:
        df = pd.read_csv(sys.argv[1], index_col="Index")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    df = df.dropna(subset=['Hogwarts House'])

    all_models = {}

    for house in df['Hogwarts House'].unique():
        x = df[INCLUDE_FEATURES].values.tolist()
        y = [1 if h == house else 0 for h in df['Hogwarts House']]
        theta0, theta, x_mean, x_std = ft_logistic_regression(x, y)
        all_models[house] = {
            "theta0": theta0,
            "theta": theta,
            "x_mean": x_mean,
            "x_std": x_std
        }
        print(f"\033[92mModel trained for house: {house}\033[0m") 
    
    print("\033[92mSaving all models to all_thetas.json\033[0m") 
    with open("all_thetas.json", "w") as f:
        json.dump(all_models, f, indent=2)

