# from simulib.config.config import kwargs


# print(kwargs)


# def a(hey, **kwargs):
#     print(hey)


# b = {"hey": 2, "five": 3}

# a(**b)

import numpy as np
import matplotlib.pyplot as plt

def plot_capex_vs_ppa_with_histogram(capex_values, demanded_ppa, breakeven_ppas, bins=30):
    """
    Create a combined line graph and histogram showing CAPEX vs. PPA prices with a histogram of breakeven PPA prices.

    :param capex_values: Array of CAPEX values for the wind contractor.
    :param demanded_ppa: Array of minimum PPA prices demanded by the wind contractor.
    :param breakeven_ppas: Array of breakeven PPA prices from the Monte Carlo simulation (1D array).
    :param bins: Number of bins for the histogram. Default is 30.
    """
    # Calculate the histogram data for breakeven PPA prices
    hist_counts, hist_bins = np.histogram(breakeven_ppas, bins=bins)

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot the CAPEX vs demanded PPA line
    ax1.plot(capex_values, demanded_ppa, label="Demanded PPA (Wind Contractor)", color="blue", linewidth=2)
    ax1.set_xlabel("CAPEX (mNOK)")
    ax1.set_ylabel("PPA Price (mNOK)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Add a secondary x-axis for the histogram
    ax2 = ax1.twiny()

    # Plot the histogram
    ax2.hist(breakeven_ppas, bins=bins, orientation='horizontal', color="grey", alpha=0.5, edgecolor="black")

    # Scale the top x-axis to fit the histogram frequency
    ax2.set_xlim(0, max(hist_counts)*3)
    ax2.set_xlabel("Frequency (Histogram)", color="grey")
    ax2.tick_params(axis='x', labelcolor="grey")

    # Add a legend to differentiate the lines
    ax1.legend(loc="upper left")

    # Set the title and show the plot
    plt.title("CAPEX vs PPA Price with Breakeven PPA Distribution")
    plt.tight_layout()
    plt.show()

# Example usage
capex_values = np.linspace(500, 2000, 10)  # CAPEX values
demanded_ppa = 0.4 * capex_values + 100  # Example linear relationship for demanded PPA
breakeven_ppas = np.random.normal(500, 100, 1000)  # Simulated breakeven PPA prices

# Call the function
plot_capex_vs_ppa_with_histogram(capex_values, demanded_ppa, breakeven_ppas, bins=30)
