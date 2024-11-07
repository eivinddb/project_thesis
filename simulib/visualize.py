import matplotlib.pyplot as plt
import numpy as np
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
    fig, axes = plt.subplots(num_state_variables, 1, figsize=(10, 3 * num_state_variables))
    
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
    Plot the distribution of NPVs from all simulation paths and mark the expected value,
    along with the average values for bins less than and greater than the specified range.
    
    :param monte_carlo_simulation: An instance of the MonteCarlo class containing the simulation results.
    :param r: Discount rate (as a decimal, e.g., 0.08 for 8%).
    """
    npvs = []
    for path in monte_carlo_simulation.paths:
        npv = net_present_value(path.cash_flows, r)
        npvs.append(npv)
    
    # Calculate the expected value (mean NPV)
    expected_value = np.mean(npvs)
    
    # Set up reasonable bins based on the known project value range
    bins = np.linspace(-4000, 12000, 100)  # 100 bins between -4000 and 12000 mNOK

    # Identify the NPVs less than -4000 and greater than 12000
    npvs_less_than_range = [npv for npv in npvs if npv < -4000]
    npvs_greater_than_range = [npv for npv in npvs if npv > 12000]

    # Calculate the average values for NPVs outside the range
    avg_less_than_range = np.mean(npvs_less_than_range) if npvs_less_than_range else None
    avg_greater_than_range = np.mean(npvs_greater_than_range) if npvs_greater_than_range else None

    # Plot the histogram of NPVs
    plt.figure(figsize=(10, 6))
    plt.hist(npvs, bins=bins, edgecolor="black", alpha=0.7)
    
    # Add a vertical line for the expected value (mean NPV)
    plt.axvline(expected_value, color='red', linestyle='dashed', linewidth=2, label=f"Expected Value: {expected_value:.2f} mNOK")
    
    # Add annotations for the average values of NPVs outside the range
    if avg_less_than_range is not None:
        plt.annotate(f"Avg. NPV < -4000: {avg_less_than_range:.2f} mNOK", 
                     xy=(-4000, 30), xycoords='data', color='green', fontsize=10, 
                     horizontalalignment='right', verticalalignment='bottom')
    
    if avg_greater_than_range is not None:
        plt.annotate(f"Avg. NPV > 12000: {avg_greater_than_range:.2f} mNOK", 
                     xy=(12000, 30), xycoords='data', color='blue', fontsize=10, 
                     horizontalalignment='left', verticalalignment='bottom')

    # Optionally, you can also include a legend to display the expected value label
    plt.legend(loc="best")
    
    plt.title("NPV Distribution of Cash Flows")
    plt.xlabel("Net Present Value (mNOK)")
    plt.ylabel("Frequency")
    plt.grid(True)
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
    lower_percentile = np.percentile(npvs, 5)
    upper_percentile = np.percentile(npvs, 85)
    
    # Create a boxplot
    plt.figure(figsize=(10, 6))
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