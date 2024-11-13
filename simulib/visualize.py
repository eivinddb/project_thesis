import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .utils import net_present_value  # Assuming this function is correctly implemented in utils.py



def plot_state_variables(monte_carlo_simulation):
    """
    Plot the evolution of state variables over time for each simulation path in separate subplots.

    :param monte_carlo_simulation: An instance of the MonteCarlo class containing the simulation results.
    """
    # Get the state variables from the first path to determine the number of subplots needed
    first_path = monte_carlo_simulation.paths[0]
    state_variables = first_path.state_variables
    num_state_variables = len(state_variables)

    # Create a figure with subplots (one for each state variable)
    fig, axes = plt.subplots(num_state_variables, 1, figsize=(10, 4 * num_state_variables))
    
    # Ensure axes is iterable (if there's only one subplot, it's a single Axes object, not an array)
    if num_state_variables == 1:
        axes = [axes]

    # Loop through each simulation path and plot each state variable on its own subplot
    for i, (name, values) in enumerate(state_variables.items()):
        ax = axes[i]
        # Use a single color for all paths
        for path in monte_carlo_simulation.paths:
            state_variables_path = path.state_variables
            ax.plot(state_variables_path[name], color='blue')  # Set the color to blue for all lines
        
        ax.set_title(f"Evolution of {name}")
        ax.set_xlabel("Time (Years)")
        ax.set_ylabel(f"{name} Value")
        ax.grid(True)
    
    plt.tight_layout()  # Adjust subplots for better spacing
    plt.show()

def plot_cash_flows(monte_carlo_simulation):
    """
    Plot the cash flows over time for each simulation path on the same plot.

    :param monte_carlo_simulation: An instance of the MonteCarlo class containing the simulation results.
    """
    plt.figure(figsize=(10, 6))
    
    # Loop through each simulation path and plot its cash flow
    for path in monte_carlo_simulation.paths:
        plt.plot(path.cash_flows, color='blue')  # Set the color to blue for all lines
    
    plt.title("Cash Flows for All Simulation Paths")
    plt.xlabel("Time (Years)")
    plt.ylabel("Cash Flow (mNOK)")
    plt.grid(True)
    plt.show()


def plot_npv_distribution(monte_carlo_simulation, r):
    """
    Plot the distribution of NPVs from all simulation paths, marking the expected value
    and the average values for bins less than and greater than the specified range.

    :param monte_carlo_simulation: An instance of the MonteCarlo class containing the simulation results.
    :param r: Discount rate (as a decimal, e.g., 0.08 for 8%).
    """
    # Calculate NPVs from the simulation paths
    npvs = [net_present_value(path.cash_flows, r) for path in monte_carlo_simulation.paths]
    
    # Compute statistics for the distribution
    expected_value = np.mean(npvs)
    percentiles = np.percentile(npvs, [10, 50, 90])
    
    # Set up bins for histogram within the fixed x-axis range
    bins = np.linspace(-5000, 12500, 30)  # Fixed range for consistent visualization
    
    # Plot histogram with KDE overlay
    plt.figure(figsize=(12, 7))
    sns.histplot(npvs, bins=bins, kde=True, color="skyblue", edgecolor="black", alpha=0.6)
    
    # Expected value line
    plt.axvline(expected_value, color="red", linestyle="--", linewidth=2, label=f"Expected Value: {expected_value:.2f} mNOK")
    
    # Add percentile markers
    plt.axvline(percentiles[0], color="purple", linestyle=":", linewidth=1.5, label=f"10th Percentile: {percentiles[0]:.2f} mNOK")
    plt.axvline(percentiles[1], color="green", linestyle=":", linewidth=1.5, label=f"Median (50th Percentile): {percentiles[1]:.2f} mNOK")
    plt.axvline(percentiles[2], color="orange", linestyle=":", linewidth=1.5, label=f"90th Percentile: {percentiles[2]:.2f} mNOK")
    
    # Set fixed x-axis limits
    plt.xlim(-5000, 15500)
    
    # Customize plot appearance
    plt.title("NPV Distribution of Cash Flows from Monte Carlo Simulation (Fixed x-axis)")
    plt.xlabel("Net Present Value (mNOK)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="best")
    
    plt.show()


def plot_npv_boxplot(monte_carlo_simulation, r):
    """
    Plot a boxplot of NPVs from all simulation paths, focusing on the quartiles for detail.
    
    :param monte_carlo_simulation: An instance of the MonteCarlo class containing the simulation results.
    :param r: Discount rate (as a decimal, e.g., 0.08 for 8%).
    """
    # Calculate NPVs for each simulation path
    npvs = [net_present_value(path.cash_flows, r) for path in monte_carlo_simulation.paths]
    
    # Calculate percentiles for more detailed view on quartiles
    lower_percentile = -5000 # np.percentile(npvs, 0)
    upper_percentile = 15000 # np.percentile(npvs, 99)
    
    # Create a boxplot
    plt.figure(figsize=(16, 6))
    plt.boxplot(npvs, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    
    # Set x-axis limits to zoom in on the central 90% of the data
    plt.xlim(lower_percentile, upper_percentile)

    # Customize plot aesthetics
    plt.title("NPV Boxplot of Cash Flows (Zoomed to Central 90%)")
    plt.xlabel("Net Present Value (mNOK)")
    plt.grid(True)
    plt.show()


def plot_state_variable_histograms_at_year(monte_carlo_simulation, year=5):
    """
    Plot histograms of state variables across all simulation paths at a specified year.
    
    :param monte_carlo_simulation: An instance of the MonteCarlo class containing the simulation results.
    :param year: The year (index) at which to plot the state variables (default is 5).
    """
    
    # Collect data for each state variable at the specified year
    state_variables_at_year = {}
    
    for path in monte_carlo_simulation.paths:
        for var_name, values in path.state_variables.items():
            if var_name not in state_variables_at_year:
                state_variables_at_year[var_name] = []
            state_variables_at_year[var_name].append(values[year])
    
    # Plot histograms for each state variable at the specified year
    num_vars = len(state_variables_at_year)
    fig, axes = plt.subplots(num_vars, 1, figsize=(10, 4 * num_vars))
    if num_vars == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot
    
    for ax, (var_name, values) in zip(axes, state_variables_at_year.items()):
        ax.hist(values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"Distribution of {var_name} at Year {year}")
        ax.set_xlabel(f"{var_name} Value")
        ax.set_ylabel("Frequency")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def improved_npv_boxplot(monte_carlo_simulation, r):
    """
    Enhanced boxplot of NPVs from all simulation paths, with median, 10th percentile, and 90th percentile values.

    :param monte_carlo_simulation: An instance of the MonteCarlo class containing the simulation results.
    :param r: Discount rate (as a decimal, e.g., 0.08 for 8%).
    """
    # Calculate NPVs for each simulation path
    npvs = [net_present_value(path.cash_flows, r) for path in monte_carlo_simulation.paths]
    
    # Calculate statistics
    median_npv = np.median(npvs)
    p10_npv = np.percentile(npvs, 10)
    p90_npv = np.percentile(npvs, 90)
    
    # Create a boxplot
    plt.figure(figsize=(12, 6))
    boxprops = dict(facecolor='lightblue', color='blue')
    medianprops = dict(color='red', linewidth=2)
    plt.boxplot(npvs, vert=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)

    # Plot additional statistics as vertical lines
    plt.axvline(median_npv, color='darkred', linestyle='--', linewidth=1.5, label=f'Median NPV: {median_npv:.2f}')
    plt.axvline(p10_npv, color='green', linestyle='--', linewidth=1.5, label=f'10th Percentile: {p10_npv:.2f}')
    plt.axvline(p90_npv, color='purple', linestyle='--', linewidth=1.5, label=f'90th Percentile: {p90_npv:.2f}')
    
    # Set plot limits for better focus on central data
    plt.xlim(p10_npv - (p90_npv - p10_npv) * 0.5, p90_npv + (p90_npv - p10_npv) * 0.5)
    
    # Customizing the plot
    plt.title("Enhanced NPV Boxplot with Median and Percentiles")
    plt.xlabel("Net Present Value (mNOK)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right")
    plt.show()


def improved_npv_distribution_zoomed(monte_carlo_simulation, r, margin=1000):
    """
    Enhanced NPV histogram with focused x-axis range around the central distribution for better readability.

    :param monte_carlo_simulation: An instance of the MonteCarlo class containing the simulation results.
    :param r: Discount rate (as a decimal, e.g., 0.08 for 8%).
    :param margin: Buffer around the 10th and 90th percentiles for x-axis limits.
    """
    # Calculate NPVs for each simulation path
    npvs = [net_present_value(path.cash_flows, r) for path in monte_carlo_simulation.paths]
    
    # Calculate statistics
    expected_value = np.mean(npvs)
    median_npv = np.median(npvs)
    p10_npv = np.percentile(npvs, 10)
    p90_npv = np.percentile(npvs, 90)
    
    # Define bins for histogram within the focused range
    bins = np.linspace(p10_npv - margin, p90_npv + margin, 30)
    
    # Plot histogram with density
    plt.figure(figsize=(12, 6))
    sns.histplot(npvs, bins=bins, kde=True, color='lightblue', edgecolor="black", alpha=0.7)
    
    # Add vertical lines for key statistics
    plt.axvline(expected_value, color='darkblue', linestyle='-', linewidth=2, label=f'Expected Value: {expected_value:.2f} mNOK')
    plt.axvline(median_npv, color='red', linestyle='--', linewidth=2, label=f'Median: {median_npv:.2f} mNOK')
    plt.axvline(p10_npv, color='green', linestyle='--', linewidth=1.5, label=f'10th Percentile: {p10_npv:.2f} mNOK')
    plt.axvline(p90_npv, color='purple', linestyle='--', linewidth=1.5, label=f'90th Percentile: {p90_npv:.2f} mNOK')
    
    # Set x-axis limits to focus on the main range
    plt.xlim(p10_npv - margin, p90_npv + margin)
    
    # Customize plot appearance
    plt.title("Enhanced NPV Distribution with Zoomed Focus, Expected Value, and Percentiles")
    plt.xlabel("Net Present Value (mNOK)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()



