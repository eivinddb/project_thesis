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
    # num_samples = min(5, state_variable_paths.shape[0])
    # for i in range(num_samples):
    #     plt.plot(years, state_variable_paths[i], color="grey", linestyle="--", alpha=0.4)

    # Customize plot
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cash_flows(years, cash_flows, title="Deterministic Cash Flows", xlabel="Year", ylabel="Cash Flow (mNOK)"):
    """
    Plot deterministic cash flows over time, distinguishing inflows and outflows.

    :param cash_flows: List or numpy array of cash flows.
    :param title: Title of the plot. Default is "Deterministic Cash Flows".
    :param xlabel: Label for the x-axis. Default is "Year".
    :param ylabel: Label for the y-axis. Default is "Cash Flow (mNOK)".
    """

    # Create the plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(years, cash_flows, color=['red' if cf < 0 else 'green' for cf in cash_flows], alpha=0.8, edgecolor='black')
    
    # Highlight cash inflows and outflows
    for bar in bars:
        height = bar.get_height()
        if height != 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, 
                     height + (5 if height > 0 else -25), 
                     f"{int(height)}", 
                     ha='center', va='bottom', fontsize=10, color='black')

    # Add gridlines for better readability
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a horizontal line at y=0
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(years)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_state_variable_boxplots(
    years, 
    state_variable_paths, 
    title="State Variable Distributions Over Time", 
    ylabel="Value (mNOK)"
):
    """
    Plot a boxplot for state variable paths for each year with x-axis scaled to the 90th percentile.

    :param years: Array of years (x-axis).
    :param state_variable_paths: 2D array of state variable paths (shape: [num_paths, len(years)]).
    :param title: Title of the plot. Default is "State Variable Distributions Over Time".
    :param ylabel: Label for the y-axis. Default is "Value (mNOK)".
    """
    # Transpose the state variable paths for boxplot
    data_by_year = state_variable_paths

    # Calculate the 90th percentile for each year
    P90 = np.percentile(state_variable_paths, 90, axis=0)
    max_90th_percentile = max(P90)

    plt.figure(figsize=(12, 6))

    # Create a boxplot for each year
    plt.boxplot(data_by_year, labels=years, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='black'),
                medianprops=dict(color='red', linewidth=2))

    # Customize plot
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.ylim(0, max_90th_percentile * 1.1)  # Scale y-axis to 10% above the maximum 90th percentile
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_breakeven_histogram(data, bins=30, title="Breakeven PPA Levels Histogram", xlabel="PPA Levels (mNOK)", ylabel="Frequency"):
    """
    Plot a histogram of breakeven PPA levels.

    :param data: List or numpy array of breakeven PPA levels.
    :param bins: Number of bins for the histogram. Default is 30.
    :param title: Title of the histogram. Default is "Breakeven PPA Levels Histogram".
    :param xlabel: Label for the x-axis. Default is "PPA Levels (mNOK)".
    :param ylabel: Label for the y-axis. Default is "Frequency".
    """
    # Calculate median for annotation
    median_value = np.median(data)

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(
        data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7,
        range=(np.min(data), np.percentile(data, 90))
    )

    # Add a vertical line for the median
    plt.axvline(median_value, color='red', linestyle='--', linewidth=2, label=f"Median: {median_value:.0f}")

    # Customize plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_histogram_with_line(
    line_x, line_y, histogram_data, bins=30,
    line_label="Line Data", histogram_label="Histogram Data",
    x_label_line="X-Axis (Line)", y_label_line="Y-Axis (Line)",
    title="Line and Histogram Plot"
):
    """
    Create a combined line graph and histogram with a secondary x-axis for the histogram,
    and add a horizontal line for the median of the histogram data.

    :param line_x: Array of x-values for the line plot.
    :param line_y: Array of y-values for the line plot.
    :param histogram_data: 1D array of data for the histogram.
    :param bins: Number of bins for the histogram. Default is 30.
    :param line_label: Label for the line data in the legend.
    :param histogram_label: Label for the histogram data in the legend.
    :param x_label_line: Label for the x-axis of the line plot.
    :param y_label_line: Label for the y-axis of the line plot.
    :param title: Title of the plot.
    """
    # Calculate the histogram data
    hist_counts, _ = np.histogram(histogram_data, bins=bins)
    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot the line graph
    ax1.plot(line_x, line_y, label=line_label, color="blue", linewidth=2)
    ax1.set_xlabel(x_label_line)
    ax1.set_ylabel(y_label_line, color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Add a secondary x-axis for the histogram
    ax2 = ax1.twiny()

    # Plot the histogram
    ax2.hist(histogram_data, bins=bins, orientation='horizontal', color="grey", 
             alpha=0.5, edgecolor="black", label=histogram_label,
             range=(min(line_y), max(line_y)))

    # Plot a horizontal line for the median of the histogram
    median_value = np.median(histogram_data)  # Calculate median of histogram data
    ax2.axhline(median_value, color="red", linestyle="--", linewidth=2, label=f"Median: {median_value:.2f}")
    p90 = np.quantile(histogram_data, 0.90)
    ax2.axhline(p90, color="grey", linestyle="--", linewidth=2, label=f"P90: {p90:.2f}")
    p10 = np.quantile(histogram_data, 0.10)
    ax2.axhline(p10, color="grey", linestyle="--", linewidth=2, label=f"P10: {p10:.2f}")

    # Scale the top x-axis to fit the histogram frequency
    ax2.set_xlim(0, max(hist_counts) * 3)
    ax2.tick_params(axis='x', labeltop=False, top=False)

    # Add a legend to differentiate the data
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Set the title and show the plot
    plt.title(title)
    plt.tight_layout()
    plt.show()
