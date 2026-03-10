import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class LayerConfig:
	"""Configuration for a single layer in the plot."""
	type : str
	label: str
	values: tuple
	color: str = "blue"

@dataclass
class PlotConfig:
	"""Configuration for the entire plot."""
	x_label: str
	y_label: str
	tittle: str
	layers: list[LayerConfig]
	output_path: str = "graph.png"

def plot(config: PlotConfig) -> None:
	"""Generates a plot based on the provided configuration.
		Args:
			config (PlotConfig): The configuration for the plot, including labels, layers, and output path.
	"""
	for layer in config.layers:
		x_data, y_data = layer.values

		if layer.type == "scatter":
			plt.scatter(x_data, y_data, label=layer.label, color=layer.color)
		elif layer.type == "line":
			plt.plot(x_data, y_data, label=layer.label, color=layer.color)
		else:
			print(f"\033[31mUnkown layer type: {layer.type}\033[0m")
		
		plt.xlabel(config.x_label)
		plt.ylabel(config.y_label)
		plt.legend()
		plt.grid(True)
		plt.savefig(config.output_path)
