# """
# Script for generating Monte Carlo simulations
# Based on Olgas epsilon paper adapted for two participants
# """

# # imports
# import numpy as np
# import math
# from simulib.montecarlo import MonteCarlo
# from simulib.process import simulate_one_gbm, simulate_one_two_factor_schwartz_smith
# from simulib.visualize import plot_gas_price_paths, plot_carbon_price_paths
# import configparser

# # Load configuration parameters
# config = configparser.ConfigParser()
# config.read("config/scenario_default.ini")

# # Retrieve scenario parameters
# params = config["scenario"]

# CAPEX_owf = float(params.get('CAPEX_owf'))
# OPEX_owf = float(params.get('OPEX_owf'))
# f_owf = float(params.get('f_owf'))
# E_pmax = float(params.get('E_pmax'))
# r = float(params.get('r'))
# T_max = int(params.get('T_max'))
# T_C = int(params.get('T_C'))
# T_LT = int(params.get('T_LT'))

# e = 2.364  # kg CO2-eq /Sm^3
# TAX_co2 = 632  # NOK /tCO2

# # agreed electricity price
# p = 0  # Set at 0 for now; can be modified as needed

# EUR_TO_NOK = 11.96 

# ##################
# # wind operator  #
# ##################
# # NOTE: currently deterministic
# class WindOperatorPath(MonteCarlo.Path):

#     def __init__(self) -> None:
#         super().__init__()

#     def simulate_state_variables(self):
#         return {}

#     def calculate_cash_flows(self):
#         E_t = np.array([0 for i in range(T_C)] + [E_pmax for i in range(T_LT - T_C)])  # electricity demand at year t
#         revenues_t = E_t * f_owf * p  # revenues related to wind electricity sales
#         CAPEX_t = np.array([CAPEX_owf / T_C for i in range(T_C)] + [0 for i in range(T_LT - T_C)])  # capex split over first two years
#         OPEX_t = np.array([0 for i in range(T_C)] + [OPEX_owf for i in range(T_LT - T_C)])  # opex equal for each operating year

#         cash_flows_wo_t = revenues_t - CAPEX_t - OPEX_t
#         return cash_flows_wo_t

# cash_flows_wo_t = WindOperatorPath().calculate_cash_flows()

# ##################
# # field operator #
# ##################
# class FieldOperatorPath(MonteCarlo.Path):

#     def __init__(self) -> None:
#         super().__init__()

#     def simulate_state_variables(self):
#         # carbon EUR / tCO2-eq -> * 11.96 # converted to NOK / tCO2-eq
#         carbon_gbm_params = {key: float(value) for key, value in config["carbon_gbm_params"].items()}
#         # gas GBP / therm -> * 5.398175 / 100 # converted to NOK / Sm^3 
#         gas_schwartz_smith_params = {key: float(value) for key, value in config["gas_schwartz_smith_params"].items()}

#         P_ets_t = simulate_one_gbm(period=T_LT, **carbon_gbm_params)
#         P_gas_t = simulate_one_two_factor_schwartz_smith(period=T_LT, **gas_schwartz_smith_params)
        
#         P_ets_t /= EUR_TO_NOK
#         P_gas_t /= EUR_TO_NOK
    
#         # Return P_ets_t and P_gas_t for further calculations and analysis
#         return {"P_ets_t": P_ets_t, "P_gas_t": P_gas_t}

#     def calculate_cash_flows(self):
#         E_t = np.array([0 for i in range(T_C)] + [E_pmax for i in range(T_LT - T_C)])
#         eta_gt = 50 / 100  # gas turbine efficiency
#         HHV_gas = 39.24  # higher heating value of gas in MJ/Sm3
#         NG_const_t = (E_t * 3.6 * 10**9) / (eta_gt * HHV_gas)  # TWh to Sm^3
#         electricity_costs_t = E_t * f_owf * p 
#         CE_t = e * NG_const_t  # carbon emissions kg CO2-eq

#         P_ets_t = self.state_variables["P_ets_t"] * 11.96  # converted to NOK / tCO2-eq
#         P_gas_t = self.state_variables["P_gas_t"] * 5.398175 / 100  # converted to NOK / Sm^3 

#         tax_co2_t = [TAX_co2]
#         Upsilon_CO2 = 2000  # NOK / tCO2

#         #TODO change to fit our data. Gradual increase in tax CO2 over initial years
#         for i in range(1, 7):
#             tax_co2_t.append(tax_co2_t[i-1] + (Upsilon_CO2 - TAX_co2) / (8 - (i + 1)))
#         tax_co2_t += [max(i, Upsilon_CO2) for i in P_ets_t[7:T_LT]]

#         CP_t = CE_t * 0.001 * tax_co2_t * 0.000001 + CE_t * 0.001 * P_ets_t * 0.000001  # converted to tCO2 and mnok
#         revenues_t = f_owf * (NG_const_t * P_gas_t * 0.000001 + CP_t)

#         CAPEX_t, OPEX_t = 0, 0
#         cash_flows_fo_t = revenues_t - CAPEX_t - OPEX_t - electricity_costs_t
#         return cash_flows_fo_t + cash_flows_wo_t

# W = 10000

# monte_carlo_simulation = MonteCarlo(FieldOperatorPath, W)
# monte_carlo_simulation.run_simulation()

# simulated_gas_paths = []
# simulated_carbon_paths = []
# for simulation in monte_carlo_simulation.paths:
#     state_variables = simulation.simulate_state_variables()
#     simulated_gas_paths.append(state_variables["P_gas_t"])
#     simulated_carbon_paths.append(state_variables["P_ets_t"])

# simulated_gas_paths = np.array(simulated_gas_paths)
# simulated_carbon_paths = np.array(simulated_carbon_paths)

# P10_gas = np.percentile(simulated_gas_paths, 10, axis=0)
# P25_gas = np.percentile(simulated_gas_paths, 25, axis=0)
# P50_gas = np.percentile(simulated_gas_paths, 50, axis=0)
# P75_gas = np.percentile(simulated_gas_paths, 75, axis=0)
# P90_gas = np.percentile(simulated_gas_paths, 90, axis=0)

# P10_carbon = np.percentile(simulated_carbon_paths, 10, axis=0)
# P25_carbon = np.percentile(simulated_carbon_paths, 25, axis=0)
# P50_carbon = np.percentile(simulated_carbon_paths, 50, axis=0)
# P75_carbon = np.percentile(simulated_carbon_paths, 75, axis=0)
# P90_carbon = np.percentile(simulated_carbon_paths, 90, axis=0)

# years = np.arange(2020, 2020 + T_LT)

# def find_closest_path(percentile_line, paths):
#     deviations = np.mean((paths - percentile_line) ** 2, axis=1)
#     closest_path_index = np.argmin(deviations)
#     return paths[closest_path_index]

# P10_gas_path = find_closest_path(P10_gas, simulated_gas_paths)
# P25_gas_path = find_closest_path(P25_gas, simulated_gas_paths)
# P50_gas_path = find_closest_path(P50_gas, simulated_gas_paths)
# P75_gas_path = find_closest_path(P75_gas, simulated_gas_paths)
# P90_gas_path = find_closest_path(P90_gas, simulated_gas_paths)


# P10_carbon_path = find_closest_path(P10_carbon, simulated_carbon_paths)
# P25_carbon_path = find_closest_path(P25_carbon, simulated_carbon_paths)
# P50_carbon_path = find_closest_path(P50_carbon, simulated_carbon_paths)
# P75_carbon_path = find_closest_path(P75_carbon, simulated_carbon_paths)
# P90_carbon_path = find_closest_path(P90_carbon, simulated_carbon_paths)

# # Plot the gas and carbon price paths
# plot_gas_price_paths(
#     years, P10_gas, P25_gas, P50_gas, P75_gas, P90_gas,
#     simulated_gas_paths, P10_gas_path, P25_gas_path, P50_gas_path, P75_gas_path, P90_gas_path)

# plot_carbon_price_paths(
#     years, P10_carbon, P25_carbon, P50_carbon, P75_carbon, P90_carbon,
#     simulated_carbon_paths, P10_carbon_path, P25_carbon_path, P50_carbon_path, P75_carbon_path,
#     P90_carbon_path)
"""
Script for generating Monte Carlo simulations
Based on Olgas epsilon paper adapted for two participants
"""
"""
Script for generating Monte Carlo simulations
Based on Olgas epsilon paper adapted for two participants
"""

# imports
import numpy as np
import math
from simulib.montecarlo import MonteCarlo
from simulib.process import simulate_one_gbm, simulate_one_two_factor_schwartz_smith
from simulib.visualize import (
    plot_gas_price_paths,
    plot_carbon_price_paths,
    plot_cash_flow_paths
)
import configparser

# Load configuration parameters
config = configparser.ConfigParser()
config.read("config/scenario_default.ini")

# Retrieve scenario parameters
params = config["scenario"]

CAPEX_owf = float(params.get('CAPEX_owf'))
OPEX_owf = float(params.get('OPEX_owf'))
f_owf = float(params.get('f_owf'))
E_pmax = float(params.get('E_pmax'))
r = float(params.get('r'))
T_max = int(params.get('T_max'))
T_C = int(params.get('T_C'))
T_LT = int(params.get('T_LT'))

e = 2.364  # kg CO2-eq /Sm^3
TAX_co2 = 632  # NOK /tCO2
EUR_TO_NOK = 11.96  # Conversion rate

# agreed electricity price
p = 0  # Set at 0 for now; can be modified as needed

##################
# wind operator  #
##################
# NOTE: currently deterministic
class WindOperatorPath(MonteCarlo.Path):

    def __init__(self) -> None:
        super().__init__()

    def simulate_state_variables(self):
        return {}

    def calculate_cash_flows(self):
        E_t = np.array([0 for i in range(T_C)] + [E_pmax for i in range(T_LT - T_C)])  # electricity demand at year t
        revenues_t = E_t * f_owf * p  # revenues related to wind electricity sales
        CAPEX_t = np.array([CAPEX_owf / T_C for i in range(T_C)] + [0 for i in range(T_LT - T_C)])  # capex split over first two years
        OPEX_t = np.array([0 for i in range(T_C)] + [OPEX_owf for i in range(T_LT - T_C)])  # opex equal for each operating year

        cash_flows_wo_t = revenues_t - CAPEX_t - OPEX_t
        return cash_flows_wo_t

cash_flows_wo_t = WindOperatorPath().calculate_cash_flows()

##################
# field operator #
##################
class FieldOperatorPath(MonteCarlo.Path):

    def __init__(self) -> None:
        super().__init__()

    def simulate_state_variables(self):
        # carbon EUR / tCO2-eq (no conversion needed)
        carbon_gbm_params = {key: float(value) for key, value in config["carbon_gbm_params"].items()}
        # gas GBP / therm -> converted to EUR/Sm³
        gas_schwartz_smith_params = {key: float(value) for key, value in config["gas_schwartz_smith_params"].items()}

        P_ets_t = simulate_one_gbm(period=T_LT, **carbon_gbm_params) 
        P_gas_t = simulate_one_two_factor_schwartz_smith(period=T_LT, **gas_schwartz_smith_params) 

        return {"P_ets_t": P_ets_t, "P_gas_t": P_gas_t}

    def calculate_cash_flows(self):
        E_t = np.array([0 for i in range(T_C)] + [E_pmax for i in range(T_LT - T_C)])
        eta_gt = 50 / 100  # gas turbine efficiency
        HHV_gas = 39.24  # higher heating value of gas in MJ/Sm3
        NG_const_t = (E_t * 3.6 * 10**9) / (eta_gt * HHV_gas)  # TWh to Sm^3
        electricity_costs_t = E_t * f_owf * p 
        CE_t = e * NG_const_t  # carbon emissions kg CO2-eq

        P_ets_t = self.state_variables["P_ets_t"]
        P_gas_t = self.state_variables["P_gas_t"]

        CP_t = CE_t * 0.001 * P_ets_t  # Carbon penalty costs in EUR
        revenues_t = f_owf * (NG_const_t * P_gas_t + CP_t)  # Revenues from avoided costs and sales

        CAPEX_t, OPEX_t = 0, 0
        cash_flows_fo_t = revenues_t - CAPEX_t - OPEX_t - electricity_costs_t
        return cash_flows_fo_t

# Number of simulation paths
W = 10000

# Create and run the Monte Carlo simulation
monte_carlo_simulation = MonteCarlo(FieldOperatorPath, W)
monte_carlo_simulation.run_simulation()

# Collect the simulated gas, carbon, and cash flow paths
simulated_gas_paths = []
simulated_carbon_paths = []
simulated_cash_flows = []
for simulation in monte_carlo_simulation.paths:
    state_variables = simulation.simulate_state_variables()
    simulated_gas_paths.append(state_variables["P_gas_t"])
    simulated_carbon_paths.append(state_variables["P_ets_t"])
    simulated_cash_flows.append(simulation.calculate_cash_flows())

# Convert lists to arrays
simulated_gas_paths = np.array(simulated_gas_paths)
simulated_carbon_paths = np.array(simulated_carbon_paths)
simulated_cash_flows = np.array(simulated_cash_flows)

# Calculate percentiles for cash flows
P10_cash_flow = np.percentile(simulated_cash_flows, 10, axis=0)
P25_cash_flow = np.percentile(simulated_cash_flows, 25, axis=0)
P50_cash_flow = np.percentile(simulated_cash_flows, 50, axis=0)
P75_cash_flow = np.percentile(simulated_cash_flows, 75, axis=0)
P90_cash_flow = np.percentile(simulated_cash_flows, 90, axis=0)

# Define years for x-axis
years = np.arange(2020, 2020 + T_LT)


P10_gas = np.percentile(simulated_gas_paths, 10, axis=0)
P25_gas = np.percentile(simulated_gas_paths, 25, axis=0)
P50_gas = np.percentile(simulated_gas_paths, 50, axis=0)
P75_gas = np.percentile(simulated_gas_paths, 75, axis=0)
P90_gas = np.percentile(simulated_gas_paths, 90, axis=0)

P10_carbon = np.percentile(simulated_carbon_paths, 10, axis=0)
P25_carbon = np.percentile(simulated_carbon_paths, 25, axis=0)
P50_carbon = np.percentile(simulated_carbon_paths, 50, axis=0)
P75_carbon = np.percentile(simulated_carbon_paths, 75, axis=0)
P90_carbon = np.percentile(simulated_carbon_paths, 90, axis=0)

years = np.arange(2020, 2020 + T_LT)

def find_closest_path(percentile_line, paths):
    deviations = np.mean((paths - percentile_line) ** 2, axis=1)
    closest_path_index = np.argmin(deviations)
    return paths[closest_path_index]

P10_gas_path = find_closest_path(P10_gas, simulated_gas_paths)
P25_gas_path = find_closest_path(P25_gas, simulated_gas_paths)
P50_gas_path = find_closest_path(P50_gas, simulated_gas_paths)
P75_gas_path = find_closest_path(P75_gas, simulated_gas_paths)
P90_gas_path = find_closest_path(P90_gas, simulated_gas_paths)


P10_carbon_path = find_closest_path(P10_carbon, simulated_carbon_paths)
P25_carbon_path = find_closest_path(P25_carbon, simulated_carbon_paths)
P50_carbon_path = find_closest_path(P50_carbon, simulated_carbon_paths)
P75_carbon_path = find_closest_path(P75_carbon, simulated_carbon_paths)
P90_carbon_path = find_closest_path(P90_carbon, simulated_carbon_paths)

# # Plot cash flow simulation results
# plot_cash_flow_paths(
#     years, P10_cash_flow, P25_cash_flow, P50_cash_flow, P75_cash_flow, P90_cash_flow,
#     simulated_cash_flows
# )
# # Plot the gas and carbon price paths
# plot_gas_price_paths(
#     years, P10_gas, P25_gas, P50_gas, P75_gas, P90_gas,
#     simulated_gas_paths, P10_gas_path, P25_gas_path, P50_gas_path, P75_gas_path, P90_gas_path)

plot_carbon_price_paths(
    years, P10_carbon, P25_carbon, P50_carbon, P75_carbon, P90_carbon,
    simulated_carbon_paths, P10_carbon_path, P25_carbon_path, P50_carbon_path, P75_carbon_path,
    P90_carbon_path)