import pandas as pd

def load_csv(file_path: str) -> pd.DataFrame:
	"""
	Loads data from a specified file path in CSV format.
	
	Args:
		file_path: The path to the data file to be loaded.
	Returns:
		A pandas DataFrame containing the loaded data.
	"""

	try:
		data = pd.read_csv(file_path, index_col="Index")
		print(f"\033[92mData loaded successfully from {file_path}\033[0m")
		return data
	except FileNotFoundError:
		raise
	except pd.errors.EmptyDataError:
		raise
	except Exception as e:
		raise