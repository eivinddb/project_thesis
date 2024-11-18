"""
Script for generating Monte Carlo simulations
Based on Olgas epsilon paper adapted for two participants
"""
# imports
import numpy as np
import math
from simulib.montecarlo import MonteCarlo
from simulib.process import simulate_one_gbm, simulate_one_two_factor_schwartz_smith, simulate_one_two_factor_schwartz_smith_ALT
from simulib.visualize import *
import configparser

# Olgas exogenous parameters
config = configparser.ConfigParser()
config.read("config/scenario_default.ini")

params = config["scenario"]

CAPEX_owf = float(params.get('CAPEX_owf'))
OPEX_owf = float(params.get('OPEX_owf'))
f_owf = float(params.get('f_owf'))
E_pmax = float(params.get('E_pmax'))
r = float(params.get('r'))
T_max = int(params.get('T_max'))
T_C = int(params.get('T_C'))
T_LT = int(params.get('T_LT'))

e = 2.364 # kg CO2-eq /Sm^3
TAX_co2 = 632 # NOK /tCO2

# agreed electricity price
p = 0 #2500 # NOK/MWh =:= mNOK/TWh

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
        E_t = np.array([0 for i in range(T_C)] + [E_pmax for i in range(T_LT - T_C)]) # electricity demand at year t
        revenues_t = E_t * f_owf * p # revenues related to wind electricity sales
        CAPEX_t = np.array([CAPEX_owf/T_C for i in range(T_C)] + [0 for i in range(T_LT - T_C)]) # capex split over first two years
        OPEX_t = np.array([0 for i in range(T_C)] + [OPEX_owf for i in range(T_LT - T_C)]) # opex equal for each operating year

        cash_flows_wo_t = revenues_t - CAPEX_t - OPEX_t

        return cash_flows_wo_t

cash_flows_wo_t = WindOperatorPath.calculate_cash_flows(None)

##################
# field operator #
##################
class FieldOperatorPath(MonteCarlo.Path):

    def __init__(self) -> None:
        super().__init__()

    def simulate_state_variables(self):
        carbon_gbm_params = {key: float(value) for key, value in config["carbon_gbm_params"].items()}
        gas_schwartz_smith_params = {key: float(value) for key, value in config["gas_schwartz_smith_params"].items()}

        P_ets_t = simulate_one_gbm(period=T_LT, **carbon_gbm_params)
        P_gas_t = simulate_one_two_factor_schwartz_smith(period=T_LT, **gas_schwartz_smith_params)

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

# Number of simulation paths
W = 10000

monte_carlo_simulation = MonteCarlo(FieldOperatorPath, W)
monte_carlo_simulation.run_simulation()

simulated_gas_paths = []

for simulation in monte_carlo_simulation.paths:
    state_variables = simulation.simulate_state_variables()
    simulated_gas_paths.append(state_variables["P_gas_t"])

simulated_gas_paths = np.array(simulated_gas_paths)

P10_gas = np.percentile(simulated_gas_paths, 10, axis=0)
P50_gas = np.percentile(simulated_gas_paths, 50, axis=0)
P90_gas = np.percentile(simulated_gas_paths, 90, axis=0)

years = np.arange(2020, 2020 + T_LT)

def find_closest_path(percentile_line, paths):
    deviations = np.mean((paths - percentile_line) ** 2, axis=1)
    closest_path_index = np.argmin(deviations)
    return paths[closest_path_index]

P10_path = find_closest_path(P10_gas, simulated_gas_paths)
P50_path = find_closest_path(P50_gas, simulated_gas_paths)
P90_path = find_closest_path(P90_gas, simulated_gas_paths)



# Assuming monte_carlo_simulation is already created and run
r = 0.08  # Example discount rate (8%)

plot_gas_price_paths(years, P10_gas, P50_gas, P90_gas, simulated_gas_paths, P10_path, P50_path, P90_path)


# Visualize the results
# plot_cash_flows(monte_carlo_simulation)

# plot_state_variables(monte_carlo_simulation)
# plot_npv_distribution(monte_carlo_simulation, r)
# plot_npv_boxplot(monte_carlo_simulation, r)
# plot_state_variable_histograms_at_year(monte_carlo_simulation, 5)



