import json

def save_model(params: dict, path :str):
	"""Saves the model to a specified path in JSON format.

	Args:
		params: The model parameters to be saved.
		path: The path where the model will be saved.
	"""
	with open(path, 'w') as f:
		json.dump(params, f)
	print(f"\033[32mModel parameters saved to {path}\033[0m")

def load_model(path: str) -> dict:
	"""Loads the model from a specified path in JSON format.

	Args:
		path: The path from which the model will be loaded.

	Returns:
		The loaded model parameters as a dictionary.
	
	Raises:
		FileNotFoundError: If the specified file does not exist.
		json.JSONDecodeError: If the file is not a valid JSON format.
		Exception: For any other unexpected errors.
	"""
	try:
		with open(path, "r") as file:
			params = json.load(file)
			# print(f"\033[32mModel loaded successfully from {path}\033[0m")
			return params
	except FileNotFoundError:
		raise 
	except json.JSONDecodeError:
		raise
	except Exception as e:
		raise
