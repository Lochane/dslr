import sys
from utils import data_tools, stats_tools
import matplotlib.pyplot as plt


def prepvalues(dataset):
    cols = data_tools.to_numeric_list(dataset)
    for col in cols:
        values = dataset[col]
        mean = stats_tools.ft_mean(values)
        std = stats_tools.ft_std_dev(values)
        dataset[col] = [(v - mean) / std for v in values]
    return dataset

def get_grades(dataset, prep_dataset, house, topic):
    grades = []
    
    for i in range(len(dataset["Hogwarts House"])):
        if dataset["Hogwarts House"][i] == house:
            value = prep_dataset[topic][i]
            if value is not None:
                grades.append(value)
    return grades
    
def histogram(dataset, prep_dataset):
    topics = prep_dataset.keys()
    for topic in topics:
        plt.figure()
        plt.hist(get_grades(dataset, prep_dataset, "Gryffindor", topic),bins=10,alpha=0.5, label='Gry', color='r')
        plt.hist(get_grades(dataset, prep_dataset, "Ravenclaw", topic),bins=10,alpha=0.5, label='Rav', color='b')
        plt.hist(get_grades(dataset, prep_dataset, "Slytherin", topic),bins=10,alpha=0.5, label='Sly', color='g')
        plt.hist(get_grades(dataset, prep_dataset, "Hufflepuff", topic),bins=10,alpha=0.5, label='Huf', color='y')
        plt.legend(loc = 'upper right')
        plt.title(topic)
        plt.savefig(f"{topic}_histogram.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <filename>")
        sys.exit(1)

    dataset = data_tools.load_csv(sys.argv[1])
    prep_dataset = prepvalues(dataset)
    histogram(dataset, prep_dataset)