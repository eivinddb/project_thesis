import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .utils import net_present_value  # Assuming this function is correctly implemented in utils.py
from sklearn.linear_model import LinearRegression


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
    

    bins = np.linspace(-4000, 12000, 20)  # Adjust based on expected NPV range
    
    # Plot histogram and overlay KDE for smoother visualization

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
 
    # Add title and labels
    plt.title("NPV Distribution of Cash Flows from Monte Carlo Simulation")
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

def plot_gas_price_paths(years, P10, P25, P50, P75, P90, paths, P10_path, P25_path, P50_path, P75_path, P90_path):
    plt.figure(figsize=(10, 6))
    
    # Plot percentile lines
    plt.plot(years, P10, color='brown', label='P10', linewidth=2)
    plt.plot(years, P25, color='purple', label='P25', linewidth=2)
    plt.plot(years, P50, color='orange', label='Expected Value (P50)', linewidth=2)
    plt.plot(years, P75, color='blue', label='P75', linewidth=2)
    plt.plot(years, P90, color='green', label='P90', linewidth=2)

    # Shade area between P10 and P90
    plt.fill_between(years, P10, P90, color='lightgreen', alpha=0.3)

    # Shade area between P25 and P75
    plt.fill_between(years, P25, P75, color='lightblue', alpha=0.3)

    # Plot the closest paths to each percentile
    plt.plot(years, P10_path, color='brown', alpha=0.6, linestyle='--', linewidth=1)
    plt.plot(years, P25_path, color='purple', alpha=0.6, linestyle='--', linewidth=1)
    plt.plot(years, P50_path, color='orange', alpha=0.6, linestyle='--', linewidth=1)
    plt.plot(years, P75_path, color='blue', alpha=0.6, linestyle='--', linewidth=1)
    plt.plot(years, P90_path, color='green', alpha=0.6, linestyle='--', linewidth=1)

    # Set plot title and labels
    plt.title("Simulated Gas Price Paths with Percentiles")
    plt.xlabel("Year")
    plt.ylabel("Gas Price (EUR/Sm³)")
    plt.xticks(np.arange(min(years), max(years) + 1, 5))
    plt.ylim(0, max(P90) * 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_carbon_price_paths(years, P10, P25, P50, P75, P90, paths, P10_path, P25_path, P50_path, P75_path, P90_path):
    plt.figure(figsize=(10, 6))
    
    # Plot percentile lines
    plt.plot(years, P10, color='brown', label='P10', linewidth=2)
    plt.plot(years, P25, color='purple', label='P25', linewidth=2)
    plt.plot(years, P50, color='orange', label='Expected Value (P50)', linewidth=2)
    plt.plot(years, P75, color='blue', label='P75', linewidth=2)
    plt.plot(years, P90, color='green', label='P90', linewidth=2)

    # Shade area between P10 and P90
    plt.fill_between(years, P10, P90, color='lightgreen', alpha=0.3)

    # Shade area between P25 and P75
    plt.fill_between(years, P25, P75, color='lightblue', alpha=0.3)

    # Plot the closest paths to each percentile
    plt.plot(years, P10_path, color='brown', alpha=0.6, linestyle='--', linewidth=1)
    plt.plot(years, P25_path, color='purple', alpha=0.6, linestyle='--', linewidth=1)
    plt.plot(years, P50_path, color='orange', alpha=0.6, linestyle='--', linewidth=1)
    plt.plot(years, P75_path, color='blue', alpha=0.6, linestyle='--', linewidth=1)
    plt.plot(years, P90_path, color='green', alpha=0.6, linestyle='--', linewidth=1)

    # Set plot title and labels
    plt.title("Simulated Carbon Price Paths with Percentiles")
    plt.xlabel("Year")
    plt.ylabel("Carbon Price (EUR/tCO₂)")
    plt.xticks(np.arange(min(years), max(years) + 1, 5))
    plt.ylim(0, max(P90) * 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cash_flow_paths(
    years, 
    P10_cash_flow, 
    P25_cash_flow, 
    P50_cash_flow, 
    P75_cash_flow, 
    P90_cash_flow, 
    simulated_cash_flows
):
    plt.figure(figsize=(10, 6))

    # Plot percentiles
    plt.plot(years, P10_cash_flow, color='brown', label='P10', linewidth=2)
    plt.plot(years, P25_cash_flow, color='purple', label='P25', linewidth=2)
    plt.plot(years, P50_cash_flow, color='orange', label='Expected Value (P50)', linewidth=2)
    plt.plot(years, P75_cash_flow, color='blue', label='P75', linewidth=2)
    plt.plot(years, P90_cash_flow, color='green', label='P90', linewidth=2)

    # Fill between percentiles to show range
    plt.fill_between(years, P10_cash_flow, P90_cash_flow, color='lightgreen', alpha=0.3, label='P10-P90 Range')

    # Optionally, plot a few sample cash flow paths for variability
    num_samples = min(5, simulated_cash_flows.shape[0])  # Plot up to 5 sample paths
    for i in range(num_samples):
        plt.plot(years, simulated_cash_flows[i], color='grey', alpha=0.5, linestyle='--', linewidth=1)

    # Labels and title
    plt.title("Simulated Cash Flow Paths with Percentiles")
    plt.xlabel("Year")
    plt.ylabel("Cash Flow (MNOK)")
    plt.xticks(np.arange(min(years), max(years) + 1, 5))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_power_demand_history(historical_years, total_power):
    historical_years = np.array(historical_years)
    total_power = np.array(total_power)

    years_reshaped = historical_years.reshape(-1, 1)

    # Fit a linear model
    model = LinearRegression()
    model.fit(years_reshaped, total_power)

    # Predict future values for the next 10 years
    future_years = np.arange(2024, 2034).reshape(-1, 1)
    future_predictions = model.predict(future_years)

    # Create a linear trendline for historical data
    trendline = model.predict(years_reshaped)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(historical_years, total_power, color='blue', label='Historical Data')
    plt.plot(historical_years, trendline, color='red', label='Linear Trendline')
    plt.plot(future_years, future_predictions, color='green', linestyle='--', label='Predicted Growth (2024-2033)')

    # Labels and Title
    plt.title('Energy Consumption Growth and Prediction', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Power (MW)', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.show()