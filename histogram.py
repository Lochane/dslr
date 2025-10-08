import sys
from utils import data_tools, stats_tools
import matplotlib.pyplot as plt
import math



def prepvalues(dataset):
    cols = [c for c in data_tools.to_numeric_list(dataset) if c.lower() != "index"]
    for col in cols:
        values = [v for v in dataset[col] if v is not None]
        if not values:
            continue
        mean = stats_tools.ft_mean(values)
        std = stats_tools.ft_std_dev(values)
        if std == 0:
            continue
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
    topics = [c for c in data_tools.to_numeric_list(prep_dataset) if c.lower() != "index"]
    n = len(topics)

    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, topic in enumerate(topics):
        gry = get_grades(dataset, prep_dataset, "Gryffindor", topic)
        rav = get_grades(dataset, prep_dataset, "Ravenclaw", topic)
        sly = get_grades(dataset, prep_dataset, "Slytherin", topic)
        huf = get_grades(dataset, prep_dataset, "Hufflepuff", topic)
        if not (gry or rav or sly or huf):
            continue

        ax = axes[i]
        ax.hist(gry, bins=30, alpha=0.5, label='Gry', color='r')
        ax.hist(rav, bins=30, alpha=0.5, label='Rav', color='b')
        ax.hist(sly, bins=30, alpha=0.5, label='Sly', color='g')
        ax.hist(huf, bins=30, alpha=0.5, label='Huf', color='y')
        ax.set_title(topic)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("all_histograms.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <filename>")
        sys.exit(1)

    dataset = data_tools.load_csv(sys.argv[1])
    prep_dataset = prepvalues(dataset)
    histogram(dataset, prep_dataset)