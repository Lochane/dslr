import sys
import json
import pandas as pd
import numpy as np

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


def predict_proba(student, theta0, theta, x_mean, x_std):
    score = theta0

    for i, feature in enumerate(INCLUDE_FEATURES):
        val = student.get(feature)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = x_mean[i]

        std = x_std[i] if x_std[i] != 0 else 1
        val_norm = (val - x_mean[i]) / std
        score += theta[i] * val_norm

    return sigmoide(score)


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

    df = pd.read_csv(sys.argv[1], index_col="Index")

    models = load_all_models()

    results = []

    for idx, row in df.iterrows():
        student = row.to_dict()
        house = predict_house(student, models)
        results.append(house)

    df["Hogwarts House"] = results
    df[["Hogwarts House"]].to_csv("houses.csv")


if __name__ == "__main__":
    main()

