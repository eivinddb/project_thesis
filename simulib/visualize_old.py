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
    ylabel="Value", 
    historical_weeks=None, 
    historical_values=None,
    save_path=None, 
    show_plot=True
):
    """
    Plot state variables over time for multiple simulation paths, including P10, P50, P90, and example paths.
    Optionally includes weekly historical values.
    Outputs the plot to a file if save_path is provided, and optionally displays it in a Jupyter Notebook.
    
    :param years: Array of years (x-axis).
    :param state_variable_paths: 2D array of state variable paths (shape: [num_paths, len(years)]).
    :param title: Title of the plot. Default is "State Variable Simulation Paths".
    :param ylabel: Label for the y-axis. Default is "Value".
    :param historical_weeks: Array of weekly timestamps for historical values (optional).
    :param historical_values: Array of historical values (optional, should align with historical_weeks).
    :param save_path: File path to save the plot. Default is None (no saving).
    :param show_plot: Whether to display the plot in the Jupyter Notebook. Default is True.
    """
    # Set font sizes and line widths for a small graph
    plt.rcParams.update({
        'font.size': 10,  # Set default font size for all text
        'axes.titlesize': 10,  # Font size for titles
        'axes.labelsize': 10,  # Font size for axis labels
        'xtick.labelsize': 10,  # Font size for x-ticks
        'ytick.labelsize': 10,  # Font size for y-ticks
        'legend.fontsize': 10,  # Font size for legend
        'figure.figsize': (8.27 * 0.9, 11.69 * 0.9 * (6 / 8.27) / 2)  # A4 page with 0.9 linewidth
    })

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
    plt.figure()  # Adjusted figure size
    plt.plot(years, P10, color="brown", label="P10")
    plt.plot(years, P50, color="orange", label="P50 (Median)")
    plt.plot(years, P90, color="green", label="P90")

    # Shade between P10 and P90
    plt.fill_between(years, P10, P90, color="lightgreen", alpha=0.3)

    # Plot example paths
    plt.plot(years, P10_path, color="gray", linestyle="--", alpha=0.6, label="Example Path")
    plt.plot(years, P50_path, color="gray", linestyle="--", alpha=0.6)
    plt.plot(years, P90_path, color="gray", linestyle="--", alpha=0.6)

    # Plot historical values if provided
    if historical_weeks is not None and historical_values is not None:
        plt.plot(historical_weeks, historical_values, color="blue", label="Historical Values", marker="o", markersize=0, linestyle="-")

    # Customize plot
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.5)  # Thinner grid lines
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout(pad=0.5)  # Reduce padding for smaller figure

    # Save and/or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

    

def plot_timeline_bar(years, cash_flows, title="Deterministic Cash Flows", xlabel="Year", ylabel="Cash Flow (mNOK)"):
    """
    Plot deterministic cash flows over time as a bar chart, with labels inside the bars and adjusted for A4 page width.
    :param years: List or numpy array of years.
    :param cash_flows: List or numpy array of cash flows.
    :param title: Title of the plot. Default is "Deterministic Cash Flows".
    :param xlabel: Label for the x-axis. Default is "Year".
    :param ylabel: Label for the y-axis. Default is "Cash Flow (mNOK)".
    """
    # Configure rcParams for consistent styling using update
    plt.rcParams.update({
        'font.size': 10,  # Set default font size for all text
        'axes.titlesize': 10,  # Font size for titles
        'axes.labelsize': 10,  # Font size for axis labels
        'xtick.labelsize': 10,  # Font size for x-ticks
        'ytick.labelsize': 10,  # Font size for y-ticks
        'legend.fontsize': 10,  # Font size for legend
        'figure.figsize': (8.27 * 0.9, 11.69 * 0.9 * (6 / 8.27) / 2)  # A4 page with 0.9 linewidth
    })

    # Create the plot
    plt.figure()
    bars = plt.bar(
        years,
        cash_flows,
        color=['red' if cf < 0 else 'green' for cf in cash_flows],
        edgecolor=None,
        alpha=0.8,
        label="Cash Flow"
    )
    
    # # Add labels inside the bars
    # for bar in bars:
    #     height = bar.get_height()
    #     if height != 0:
    #         plt.text(
    #             bar.get_x() + bar.get_width() / 2.0,
    #             height - (40 if height > 0 else -40),  # Adjust label position inside the bar
    #             f"{int(height)}",
    #             ha='center',
    #             va='center',
    #             color='white',
    #             fontsize=10
    #         )
    
    # Add horizontal line at y=0
    plt.axhline(0, color='black', linewidth=1, linestyle='-')  # Add a horizontal line at y=0
    
    # Titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(years)
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def plot_line(years, cash_flows, title="Deterministic Cash Flows", xlabel="Year", ylabel="Cash Flow (mNOK)"):
    """
    Plot deterministic cash flows over time as a line plot, adjusted for A4 page width at 0.9 linewidth.
    :param years: List or numpy array of years.
    :param cash_flows: List or numpy array of cash flows.
    :param title: Title of the plot. Default is "Deterministic Cash Flows".
    :param xlabel: Label for the x-axis. Default is "Year".
    :param ylabel: Label for the y-axis. Default is "Cash Flow (mNOK)".
    """
    # Configure rcParams for consistent styling using update
    plt.rcParams.update({
        'font.size': 10,  # Set default font size for all text
        'axes.titlesize': 10,  # Font size for titles
        'axes.labelsize': 10,  # Font size for axis labels
        'xtick.labelsize': 10,  # Font size for x-ticks
        'ytick.labelsize': 10,  # Font size for y-ticks
        'legend.fontsize': 10,  # Font size for legend
        'figure.figsize': (8.27 * 0.9, 11.69 * 0.9 * (6 / 8.27) / 2)  # A4 page with 0.9 linewidth
    })

    # Create the plot
    plt.figure()
    plt.plot(years, cash_flows, marker='o', color='blue', label="Cash Flow")
    
    # # Highlight cash inflows and outflows with text
    # for x, y in zip(years, cash_flows):
    #     plt.text(x, y+40, f"{int(y)}", ha='center', va='center', color='black')
    
    # Add gridlines for better readability
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a horizontal line at y=0
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(years)
    plt.legend()
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


def plot_histogram(data, bins=30, title="Title", xlabel="(mNOK)", ylabel="Frequency"):
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


def plot_waterfall(years, cash_flows, r, title="Waterfall Plot of Discounted Cash Flows", xlabel="Year", ylabel="Value (mNOK)"):
    """
    Plot a waterfall chart of discounted cash flows over time with a total NPV column.
    :param years: List or numpy array of years.
    :param cash_flows: List or numpy array of cash flows.
    :param r: Discount rate as a decimal (e.g., 0.05 for 5%).
    :param title: Title of the plot. Default is "Waterfall Plot of Discounted Cash Flows".
    :param xlabel: Label for the x-axis. Default is "Year".
    :param ylabel: Label for the y-axis. Default is "Value (mNOK)".
    """
    # Calculate discounted cash flows
    discounted_cash_flows = [cf / ((1 + r) ** i) for i, cf in enumerate(cash_flows, start=1)]
    cumulative_values = np.cumsum([0] + discounted_cash_flows[:-1])
    total_npv = -sum(discounted_cash_flows)

    # Extend the data for the total NPV column
    years_extended = list(range(len(years))) + [len(years)]
    discounted_cash_flows_extended = discounted_cash_flows + [total_npv]
    cumulative_values_extended = np.cumsum([0] + discounted_cash_flows_extended[:-1])

    # Configure rcParams for consistent styling
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8.27 * 0.9, 11.69 * 0.9 * (6 / 8.27) / 2),
        'axes.xmargin': 0.05  # Add more space between the axis and bars horizontally
    })

    # Determine y-axis limits with padding
    min_value = min(0, *np.cumsum(discounted_cash_flows_extended))
    max_value = max(0, *np.cumsum(discounted_cash_flows_extended))
    y_padding = 0.05 * (max_value - min_value)  # Add 5% padding

    # Create the waterfall plot
    plt.figure()
    bar_colors = ['red' if cf < 0 else 'green' for cf in discounted_cash_flows] + ['black']
    bars = plt.bar(
        years_extended,
        discounted_cash_flows_extended,
        bottom=cumulative_values_extended,
        color=bar_colors,
        edgecolor=None,
        #alpha=0.8
    )

    # Add labels inside the bars
    for i, (bar, value) in enumerate(zip(bars, discounted_cash_flows_extended)):
        height = bar.get_height()
        if i == len(discounted_cash_flows_extended) - 1:  # Last column (Total NPV)
            # Place text outside the bar
            position = bar.get_y() + height / 2
            va = 'bottom' if height < 0 else 'top'
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                position,
                f"{-value:.0f}",
                ha='center',
                va=va,
                color='black',
                fontsize=10,
            )
        else:
            # Place text inside the bar
            position = bar.get_y() + height / 2
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                position,
                f"{value:.0f}",
                ha='center',
                va='center',
                color='white',
                fontsize=10
            )

    # Add horizontal lines between the bars
    for i in range(len(cumulative_values_extended) - 1):
        x_start = bars[i].get_x() #- bars[i].get_width()/2
        x_end = bars[i + 1].get_x() + bars[i+1].get_width()*0.99
        y = cumulative_values_extended[i + 1]
        plt.hlines(y, x_start, x_end, color='black', linewidth=1, linestyle='-')

    # Add a horizontal line at y=0
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    # Set y-axis limits with padding
    plt.ylim(min_value - y_padding, max_value + y_padding)

    # Titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(years_extended)), [*years, "Total"])
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_waterfall_contributors(contributors, cash_flows, title="Waterfall Plot of NPV Contributors", ylabel="Value (mNOK)"):
    """
    Plot a waterfall chart of NPV contributors.
    
    :param contributors: List of contributor names (e.g., "Capex", "Opex", "Revenue").
    :param cash_flows: List or numpy array of cash flows corresponding to each contributor.
    :param title: Title of the plot. Default is "Waterfall Plot of NPV Contributors".
    :param ylabel: Label for the y-axis. Default is "Value (mNOK)".
    """
    # Calculate cumulative values
    cumulative_values = np.cumsum([0] + cash_flows[:-1])

    # Extend contributors and cash flows for the total NPV column
    contributors_extended = contributors + ["Total"]
    cash_flows_extended = cash_flows + [-sum(cash_flows)]
    cumulative_values_extended = np.cumsum([0] + cash_flows_extended[:-1])

    # Configure rcParams for consistent styling
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (8.27 * 0.9, 11.69 * 0.9 * (6 / 8.27) / 2),
        'axes.xmargin': 0.05  # Add more space between the axis and bars horizontally
    })

    # Determine y-axis limits with padding
    min_value = min(0, *cumulative_values_extended)
    max_value = max(0, *cumulative_values_extended)
    y_padding = 0.05 * (max_value - min_value)

    # Create the waterfall plot
    plt.figure()
    bar_colors = ['red' if cf < 0 else 'green' for cf in cash_flows] + ['black']
    bars = plt.bar(
        range(len(contributors_extended)),
        cash_flows_extended,
        bottom=cumulative_values_extended,
        color=bar_colors,
        edgecolor=None,
    )

    # Add labels inside the bars
    for i, (bar, value) in enumerate(zip(bars, cash_flows_extended)):
        height = bar.get_height()
        position = bar.get_y() + height / 2
        if i == len(cash_flows_extended) - 1:  # Last column (Total NPV)
            position = bar.get_y() - height
            va = 'bottom' if height < 0 else 'top'
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                position,
                f"{-value:.0f}",
                ha='center',
                va=va,
                color='black',
                fontsize=10,
            )
        else:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                position,
                f"{value:.0f}",
                ha='center',
                va='center',
                color='white',
                fontsize=10
            )

    # Add horizontal lines between the bars
    for i in range(len(cumulative_values_extended) - 1):
        x_start = bars[i].get_x()
        x_end = bars[i + 1].get_x() + bars[i + 1].get_width() * 0.99
        y = cumulative_values_extended[i + 1]
        plt.hlines(y, x_start, x_end, color='black', linewidth=1, linestyle='-')

    # Add a horizontal line at y=0
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    # Set y-axis limits with padding
    plt.ylim(min_value - y_padding, max_value + y_padding)

    # Titles and labels
    # plt.title(title)
    # plt.xlabel("Contributors")
    plt.ylabel(ylabel)
    plt.xticks(range(len(contributors_extended)), contributors_extended, rotation=20)
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_pie_contributors_split(contributors, cash_flows, title="NPV Contributors Pie Chart"):
    """
    Plot two pie charts: one for costs (negative values) and one for profits (positive values).
    
    :param contributors: List of contributor names (e.g., "Capex", "Opex", "Revenue").
    :param cash_flows: List or numpy array of cash flows corresponding to each contributor.
    :param title: Title of the plot. Default is "NPV Contributors Pie Chart".
    """
    # Update rcParams for consistent styling
    plt.rcParams.update({
        'font.size': 10,  # Set default font size for all text
        'axes.titlesize': 10,  # Font size for titles
        'axes.labelsize': 10,  # Font size for axis labels
        'xtick.labelsize': 10,  # Font size for x-ticks
        'ytick.labelsize': 10,  # Font size for y-ticks
        'legend.fontsize': 10,  # Font size for legend
        'figure.figsize': (8.27 * 0.9, 11.69 * 0.9 * (6 / 8.27) / 2),  # A4 page with 0.9 linewidth
    })

    # Split contributors into costs and profits
    costs = [(name, cf) for name, cf in zip(contributors, cash_flows) if cf < 0]
    profits = [(name, cf) for name, cf in zip(contributors, cash_flows) if cf > 0]

    # Unpack data
    cost_labels, cost_values = zip(*costs) if costs else ([], [])
    profit_labels, profit_values = zip(*profits) if profits else ([], [])

    # Absolute values for the pie charts
    cost_values = [-value for value in cost_values]  # Convert to positive for the pie chart
    total_costs = sum(cost_values)
    total_profits = sum(profit_values)

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(8.27 * 0.9, 11.69 * 0.9 * (6 / 8.27) / 2), dpi=100)

    # Costs pie chart
    if cost_values:
        wedges, texts, autotexts = axes[0].pie(
            cost_values,
            labels=cost_labels,
            autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
            colors=["#c9daf8ff" for _ in cost_values],  # Softer red tones
            startangle=90,
            textprops={'fontsize': 10},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.8},  # Add borders
        )
        axes[0].set_title(f"Costs (Total: {total_costs:.1f} mNOK)", fontsize=10)

    # Profits pie chart
    if profit_values:
        wedges, texts, autotexts = axes[1].pie(
            profit_values,
            labels=profit_labels,
            autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
            colors=["#c9daf8ff" for _ in profit_values],  # Softer green tones
            startangle=90,
            textprops={'fontsize': 10},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.8},  # Add borders
        )
        axes[1].set_title(f"Profits (Total: {total_profits:.1f} mNOK)", fontsize=10)

    # Overall title
    # fig.suptitle(title, fontsize=12)

    # Adjust spacing and layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.3)

    # Show the plot
    plt.show()