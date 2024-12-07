import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .utils import net_present_value  # Assuming this function is correctly implemented in utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_state_variable_paths(
    years, 
    state_variable_paths, 
    title="State Variable Simulation Paths", 
    ylabel="Value"
):
    """
    Plot state variables over time for multiple simulation paths, including P10, P50, P90, and example paths.

    :param years: Array of years (x-axis).
    :param state_variable_paths: 2D array of state variable paths (shape: [num_paths, len(years)]).
    :param title: Title of the plot. Default is "State Variable Simulation Paths".
    :param ylabel: Label for the y-axis. Default is "Value".
    """
    # Calculate percentiles
    P10 = np.percentile(state_variable_paths, 10, axis=0)
    P50 = np.percentile(state_variable_paths, 50, axis=0)
    P90 = np.percentile(state_variable_paths, 90, axis=0)

    # Find example paths (closest to percentiles)
    def find_closest_path(percentile_line):
        deviations = np.mean((state_variable_paths - percentile_line) ** 2, axis=1)
        closest_path_index = np.argmin(deviations)
        return state_variable_paths[closest_path_index]

    P10_path = find_closest_path(P10)
    P50_path = find_closest_path(P50)
    P90_path = find_closest_path(P90)

    # Plot percentiles
    plt.figure(figsize=(12, 6))
    plt.plot(years, P10, color="brown", label="P10", linewidth=2)
    plt.plot(years, P50, color="orange", label="P50 (Median)", linewidth=2)
    plt.plot(years, P90, color="green", label="P90", linewidth=2)

    # Shade between P10 and P90
    plt.fill_between(years, P10, P90, color="lightgreen", alpha=0.3, label="P10-P90 Range")

    # Plot example paths
    plt.plot(years, P10_path, color="brown", linestyle="--", alpha=0.6, label="Example Path (P10)")
    plt.plot(years, P50_path, color="orange", linestyle="--", alpha=0.6, label="Example Path (P50)")
    plt.plot(years, P90_path, color="green", linestyle="--", alpha=0.6, label="Example Path (P90)")

    # Optionally, plot a few random sample paths
    num_samples = min(5, state_variable_paths.shape[0])
    for i in range(num_samples):
        plt.plot(years, state_variable_paths[i], color="grey", linestyle="--", alpha=0.4)

    # Customize plot
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
