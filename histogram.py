import sys
from utils import data_tools, stats_tools
import matplotlib.pyplot as plt


def prepvalues(dataset):
    """Normalise uniquement les colonnes numériques, en ignorant les None et std=0."""
    cols = data_tools.to_numeric_list(dataset)
    for col in cols:
        values = [v for v in dataset[col] if v is not None]
        if not values:
            continue
        mean = stats_tools.ft_mean(values)
        # std basé sur les valeurs valides uniquement
        std = stats_tools.ft_std_dev(values)
        if std == 0:
            # Evite division par zéro: laisse la colonne telle quelle
            continue
        # Applique la normalisation, en gardant None pour les cellules vides
        norm = []
        it = dataset[col]
        for v in it:
            if v is None:
                norm.append(None)
            else:
                norm.append((v - mean) / std)
        dataset[col] = norm
    return dataset

def get_grades(dataset, prep_dataset, house, topic):
    grades = []
    houses = dataset.get("Hogwarts House")
    if houses is None:
        return grades
    for i in range(len(houses)):
        if houses[i] == house:
            value = prep_dataset[topic][i]
            if value is not None:
                grades.append(value)
    return grades
    
def histogram(dataset, prep_dataset):
    topics = [c for c in data_tools.to_numeric_list(prep_dataset)]
    for topic in topics:
        gry = get_grades(dataset, prep_dataset, "Gryffindor", topic)
        rav = get_grades(dataset, prep_dataset, "Ravenclaw", topic)
        sly = get_grades(dataset, prep_dataset, "Slytherin", topic)
        huf = get_grades(dataset, prep_dataset, "Hufflepuff", topic)
        if not (gry or rav or sly or huf):
            continue
        plt.figure()
        plt.hist(gry, bins=30, alpha=0.5, label='Gry', color='r')
        plt.hist(rav, bins=30, alpha=0.5, label='Rav', color='b')
        plt.hist(sly, bins=30, alpha=0.5, label='Sly', color='g')
        plt.hist(huf, bins=30, alpha=0.5, label='Huf', color='y')
        plt.legend(loc='upper right')
        plt.title(topic)
        plt.tight_layout()
        plt.savefig(f"{topic}_histogram.png")
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <filename>")
        sys.exit(1)

    dataset = data_tools.load_csv(sys.argv[1])
    prep_dataset = prepvalues(dataset)
    histogram(dataset, prep_dataset)