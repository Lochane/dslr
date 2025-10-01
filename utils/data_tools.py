import csv

def load_csv(path):
	with open(path, newline='') as file:
		reader = csv.DictReader(file)
		data = {col: [] for col in reader.fieldnames}
		for row in reader:
			for col, val in row.items():
				try:
					data[col].append(float(val))
				except ValueError:
					pass
		return data
	
def to_numeric_list(dataset):
    numeric_col = []
    for col_name, values in dataset.items():
        if not values:
            continue
        try:
            float(values[0])
            numeric_col.append(col_name)
        except (ValueError, TypeError):
            continue
    return numeric_col