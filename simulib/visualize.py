import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .utils import net_present_value  # Assuming this function is correctly implemented in utils.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

def plot_state_variable_paths(
    years, 
    state_variable_paths, 
    ylabel="Value", 
    historical_weeks=None, 
    historical_values=None,
    save_path=None, 
    show_plot=True
):
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

    
    plt.figure()  # Adjusted figure size

    # Add a horizontal line at y=0
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    # Plot percentiles
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
    # plt.title(title)
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

    
def plot_contributors_pie(
        contributors, 
        cash_flows, 
        color=None, 
        title="NPV Contributors Pie Chart", 
        save_path=None, 
        show_plot=True
):
    plt.rcParams.update({
        'font.size': 10,  # Set default font size for all text
        'axes.titlesize': 10,  # Font size for titles
        'axes.labelsize': 10,  # Font size for axis labels
        'xtick.labelsize': 10,  # Font size for x-ticks
        'ytick.labelsize': 10,  # Font size for y-ticks
        'legend.fontsize': 10,  # Font size for legend
        'figure.figsize': (8.27 * 0.45, 11.69 * 0.9 * (6 / 8.27) / 2)  # A4 page with 0.9 linewidth
    })

    # Convert all cash flows to absolute values for the pie chart
    absolute_values = [abs(cf) for cf in cash_flows]

    # Assign colors based on whether the value is a cost or a profit
    if color:
        colors = [color for cf in cash_flows]
    else:
        colors = ["red" if cf < 0 else "green" for cf in cash_flows]

    # Create the pie chart
    plt.figure()
    wedges, texts, autotexts = plt.pie(
        absolute_values,
        labels=contributors,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10, 'color':'black'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8},  # Add borders
    )

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Show the plot if required
    if show_plot:
        plt.show()

    # Clear the figure to free up memory
    plt.close()


def plot_contributors_waterfall(
        contributors, 
        cash_flows, 
        ylabel="Value (mNOK)",
        save_path=None, 
        show_plot=True
):
    
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

    # Add a horizontal line at y=0
    plt.axhline(0, color='black', linewidth=1, linestyle='--')


    bar_colors = ['red' if cf < 0 else 'green' for cf in cash_flows] + ['black']
    bars = plt.bar(
        range(len(contributors_extended)),
        cash_flows_extended,
        bottom=cumulative_values_extended,
        color=bar_colors,
        edgecolor=None,
    )

    # Calculate text height in data coordinates
    ax = plt.gca()
    renderer = plt.gcf().canvas.get_renderer()
    sample_text = ax.text(0, 0, "0")
    text_height = sample_text.get_window_extent(renderer=renderer).transformed(ax.transData.inverted()).height
    sample_text.remove()  # Cleanup

    cash_flows_extended[-1] = -cash_flows_extended[-1]
    # Add labels dynamically based on bar height
    for i, (bar, value) in enumerate(zip(bars, cash_flows_extended)):
        height = bar.get_height()

        down_bool = (bar.get_y() < 0) or (bar.get_y() == max_value)

        # Decide if text should go inside or outside the bar
        if abs(height) > 0.05*(max_value - min_value):  # Check if the text fits inside the bar
            va = 'center'
            position = bar.get_y() + height / 2
            color = 'white'
        else:  # Place the text above or below the bar
            va = 'bottom' if not down_bool else 'top'
            position = bar.get_y() + (-1 if down_bool else 1) * 0.02*(max_value - min_value)
            color = 'black'

        # Add text
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            position,
            f"{value:.0f}",
            ha='center',
            va=va,
            color=color,
        )
    cash_flows_extended[-1] = -cash_flows_extended[-1]

    # Add horizontal lines between the bars
    for i in range(len(cumulative_values_extended) - 1):
        x_start = bars[i].get_x()
        x_end = bars[i + 1].get_x() + bars[i + 1].get_width() * 0.99
        y = cumulative_values_extended[i + 1]
        plt.hlines(y, x_start, x_end, color='black', linewidth=1, linestyle='-')

    
    # Set y-axis limits with padding
    plt.ylim(min_value - y_padding, max_value + y_padding)

    # Titles and labels
    plt.ylabel(ylabel)
    plt.xticks(range(len(contributors_extended)), contributors_extended, rotation=20)
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Show the plot if required
    if show_plot:
        plt.show()

    # Clear the figure to free up memory
    plt.close()


def plot_timeline_bar(
        years, 
        cash_flows, 
        xlabel="Year", 
        ylabel="Cash Flow (mNOK)",
        save_path=None, 
        show_plot=True
):
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
    
    # Add labels inside the bars
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.05*(max(cash_flows) - min(cash_flows)):  # Check if the text fits inside the bar
            va = 'center'
            position = height / 2
            color = 'white'
        else:  # Place the text above or below the bar
            va = 'bottom' if height > 0 else 'top'
            position = height + (-1 if height < 0 else 1) * 0.01*(max(cash_flows) - min(cash_flows))
            color = 'black'
        if height != 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                position, #height - (40 if height > 0 else -40),  # Adjust label position inside the bar
                f"{int(height)}",
                ha='center',
                va=va,
                color=color,
            )

    
    # Add horizontal line at y=0
    plt.axhline(0, color='black', linewidth=1, linestyle='-')  # Add a horizontal line at y=0
    
    # Titles and labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(years)
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Show the plot if required
    if show_plot:
        plt.show()

    # Clear the figure to free up memory
    plt.close()


def plot_timeline_line(
        years, 
        cash_flows, 
        legend_label="Cash Flow", ylabel="Cash Flow (mNOK)",
        save_path=None, 
        show_plot=True
):
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
    plt.plot(years, cash_flows, marker='o', color='blue')#, label=legend_label)
    
    # # Highlight cash inflows and outflows with text
    # for x, y in zip(years, cash_flows):
    #     plt.text(x, y+40, f"{int(y)}", ha='center', va='center', color='black')
    
    # Add gridlines for better readability
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # Add a horizontal line at y=0
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Titles and labels
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.xticks(years)
    # plt.legend()
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Show the plot if required
    if show_plot:
        plt.show()

    # Clear the figure to free up memory
    plt.close()


def plot_histogram(
        data, 
        bins=30, 
        xlabel="(mNOK)", 
        ylabel="Frequency",
        max_percentile_limit = 100,
        percentile_lines = [],
        width=1,
        save_path=None, 
        show_plot=True
):
    
    # Configure A4 styling
    plt.rcParams.update({
        'font.size': 10,  # Set default font size for all text
        'axes.titlesize': 10,  # Font size for titles
        'axes.labelsize': 10,  # Font size for axis labels
        'xtick.labelsize': 10,  # Font size for x-ticks
        'ytick.labelsize': 10,  # Font size for y-ticks
        'legend.fontsize': 10,  # Font size for legend
        'figure.figsize': (8.27 * 0.9 * width, 11.69 * 0.9 * (6 / 8.27) / 2),  # A4 dimensions with scaling
        'axes.xmargin': 0.05  # Add small horizontal margin
    })

    # Calculate median for annotation
    median_value = np.median(data)

    # Plot histogram
    plt.figure()
    plt.hist(
        data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7,
        range=(np.min(data), np.percentile(data, max_percentile_limit))
    )

    # Add a vertical line for the median
    plt.axvline(median_value, color='red', linestyle='--', linewidth=2, label=f"Median: {median_value:.0f}")

    for percentile in percentile_lines:
        percentile_value = np.percentile(data, percentile)
        plt.axvline(percentile_value, color='gray', linestyle='--', linewidth=2, label=f"{percentile}th percentile: {percentile_value:.0f}")

    # Customize plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Show the plot if required
    if show_plot:
        plt.show()

    # Clear the figure to free up memory
    plt.close()