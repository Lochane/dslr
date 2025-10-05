import csv

def load_csv(path):
	with open(path, newline='') as file:
		reader = csv.DictReader(file)
		data = {col: [] for col in reader.fieldnames}
		for row in reader:
			for col, val in row.items():
				if val is None or val == "":
					data[col].append(None)
					continue
				try:
					data[col].append(float(val))
				except (ValueError, TypeError):
					data[col].append(val)
		return data

def to_numeric_list(dataset):
	numeric_col = []
	for col_name, values in dataset.items():
		sample = None
		for v in values:
			if v is not None:
				sample = v
				break
		if sample is None:
			continue
		try:
			float(sample)
			numeric_col.append(col_name)
		except (ValueError, TypeError):
			continue
	return numeric_col