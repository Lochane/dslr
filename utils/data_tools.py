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
	
def to_numeric_list(values):
    nums = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() == "nan":
                continue
        try:
            nums.append(float(v))
        except Exception:
            continue
    return nums

def is_numeric_column(values, threshold=0.9, min_numeric=2):
    non_empty = 0
    numeric = 0
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() == "nan":
                continue
        non_empty += 1
        try:
            float(v)
            numeric += 1
        except Exception:
            pass
    if non_empty == 0:
        return False
    return numeric >= min_numeric and (numeric / non_empty) >= threshold