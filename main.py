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
    plot_cash_flow_paths, plot_power_demand_history
)
import configparser
from sklearn.linear_model import LinearRegression

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
TAX_co2 = 790  # NOK /tCO2, changed from 2023 rate to 2024 rate
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
        carbon_gbm_params = {key: float(value) for key, value in config["carbon_gbm_params"].items()}
        # carbon_schwartz_smith_params = {key: float(value) for key, value in config["carbon_schwartz_smith_params"].items()}
        gas_schwartz_smith_params = {key: float(value) for key, value in config["gas_schwartz_smith_params"].items()}

        # Generate gas price in GBP/therm and convert to EUR/Sm³
        P_gas_t = simulate_one_two_factor_schwartz_smith(period=T_LT, **gas_schwartz_smith_params)
        # GBP_TO_EUR = 1.15 
        # THERM_TO_SM3 = 2.83
        # P_gas_t = P_gas_t * (1 / THERM_TO_SM3) * GBP_TO_EUR

        # Generate carbon price (no conversion needed as already in EUR/tCO2)
        P_ets_t = simulate_one_gbm(period=T_LT, **carbon_gbm_params) 

        return {"P_ets_t": P_ets_t, "P_gas_t": P_gas_t}

    def calculate_cash_flows(self):
        E_t = np.array([0 for i in range(T_C)] + [E_pmax for i in range(T_LT - T_C)])
        eta_gt = 50 / 100 # % https://www.ipieca.org/resources/energy-efficiency-compendium-online/open-cycle-gas-turbines-2022#:~:text=The%20efficiency%20of%20an%20OCGT,drops%20significantly%20at%20partial%20load).
        HHV_gas = 39.24 # MJ/Sm3 https://en.wikipedia.org/wiki/Heat_of_combustion#Higher_heating_values_of_natural_gases_from_various_sources
        NG_const_t = (E_t * 3.6 * 10**9) / (eta_gt * HHV_gas) # 3.6 * 10**9 is to convert TWh to MJ - Sm^3
        electricitycosts_t = E_t * f_owf * p 
        CE_t = e * NG_const_t # carbon emissions kg CO2-eq

        P_ets_t = self.state_variables["P_ets_t"] * 11.96 # converted to NOK / tCO2-eq
        P_gas_t = self.state_variables["P_gas_t"] * 5.398175 / 100 # converted to NOK / Sm^3 

        tax_co2_t = [TAX_co2]
        Upsilon_CO2 = 2000 # NOK / tCO2

        #TODO Fix this to fit our period
        for i in range(1, 7):
            tax_co2_t.append(tax_co2_t[i-1] + (Upsilon_CO2 - TAX_co2)/8-(i+1))
        tax_co2_t += [max(i,Upsilon_CO2)  for i in P_ets_t[7:T_LT]]

        CP_t = CE_t * 0.001 * tax_co2_t * 0.000001 + CE_t * 0.001 * P_ets_t * 0.000001 # converted to tCO2 and mnok

        revenues_t = f_owf*(NG_const_t * P_gas_t * 0.000001 + CP_t)

        CAPEX_t = 0
        OPEX_t = 0

        # TODO: electricity costs/electricity purchase expenses (from paper not below) --> ASK OLGA
        cash_flows_fo_t = revenues_t - CAPEX_t - OPEX_t - electricitycosts_t

        return cash_flows_fo_t + cash_flows_wo_t


##################
# MC Simulation #
##################
W = 10000

monte_carlo_simulation = MonteCarlo(FieldOperatorPath, W)
monte_carlo_simulation.run_simulation()

simulated_gas_paths = []
simulated_carbon_paths = []
simulated_cash_flows = []
for simulation in monte_carlo_simulation.paths:
    state_variables = simulation.simulate_state_variables()
    simulated_gas_paths.append(state_variables["P_gas_t"])
    simulated_carbon_paths.append(state_variables["P_ets_t"])
    simulated_cash_flows.append(simulation.calculate_cash_flows())

simulated_gas_paths = np.array(simulated_gas_paths)
simulated_carbon_paths = np.array(simulated_carbon_paths)
simulated_cash_flows = np.array(simulated_cash_flows)

P10_cash_flow = np.percentile(simulated_cash_flows, 10, axis=0)
P25_cash_flow = np.percentile(simulated_cash_flows, 25, axis=0)
P50_cash_flow = np.percentile(simulated_cash_flows, 50, axis=0)
P75_cash_flow = np.percentile(simulated_cash_flows, 75, axis=0)
P90_cash_flow = np.percentile(simulated_cash_flows, 90, axis=0)

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

years = np.arange(2024, 2024 + T_LT)

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

# plot_gas_price_paths(
#     years, P10_gas, P25_gas, P50_gas, P75_gas, P90_gas,
#     simulated_gas_paths, P10_gas_path, P25_gas_path, P50_gas_path, P75_gas_path, P90_gas_path)

plot_carbon_price_paths(
    years, P10_carbon, P25_carbon, P50_carbon, P75_carbon, P90_carbon,
    simulated_carbon_paths, P10_carbon_path, P25_carbon_path, P50_carbon_path, P75_carbon_path, P90_carbon_path)


##################
# Power demand #
##################
historical_years = [
    1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
    2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023
]
total_power = [
    0.4, 11.3, 32.8, 65.0, 98.9, 140.4, 135.3, 125.5, 137.1, 165.2, 177.5, 179.8, 199.5, 175.2, 217.4, 332.7, 396.3, 411.7,
    337.7, 261.1, 217.6, 210.0, 291.5, 346.2, 268.1, 270.8, 318.7, 300.8, 270.5, 266.9, 474.5
]
# plot_power_demand_history(historical_years, total_power)
