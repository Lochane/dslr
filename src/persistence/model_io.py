import json
import os
def save_model(params: dict, filepath :str):
		"""Creates the directory for the model file and saves the model to a specified path in JSON format.

		Args:
				params: The model parameters to be saved.
				filepath: The path where the model will be saved.
		"""
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		print(f"Directory '{filepath}' created successfully.")
		with open(filepath, 'w') as f:
				json.dump(params, f,indent=2)
				print(f"\033[32mModel parameters saved to {filepath}\033[0m")

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
