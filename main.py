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

# Olgas exogenous parameters
CAPEX_owf = 3500 # mNOK
OPEX_owf = 200 # mNOK
TAX_co2 = 632 # NOK /tCO2
r = 8 / 100 # %
T_max = 10 # years # NOTE: decision period not yet usable
T_C = 2 # years # TODO: unused
T_LT = 25 # years
f_owf = 35 / 100 # %
e = 2.364 # kg CO2-eq /Sm^3
E_pmax = 0.3 # TWh


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
        E_t = np.array([0, 0] + [E_pmax for i in range(T_LT - 2)]) # electricity demand at year t
        revenues_t = E_t * f_owf * p # revenues related to wind electricity sales
        CAPEX_t = np.array([CAPEX_owf/2, CAPEX_owf/2] + [0 for i in range(T_LT - 2)]) # capex split over first two years
        OPEX_t = np.array([0, 0] + [OPEX_owf for i in range(T_LT - 2)]) # opex equal for each operating year

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
        # carbon EUR / tCO2-eq -> * 11.96 # converted to NOK / tCO2-eq
        carbon_gbm_params = {
            "mu": 0.17,
            "sigma": 0.386,
            "p_0": 97.1,
            "dt": 1,
            "period": T_LT
        }

        # gas GBP / therm -> * 5.398175 / 100 # converted to NOK / Sm^3 
        gas_schwartz_smith_params = {
            "dt": 1,
            "period": T_LT,

            "xi_0": 4.38,
            "sigma_xi": 0.24,
            "mu_xi": -0.04,
            "kappa": 0.89,

            "chi_0": 0,
            "sigma_chi": 0.68,
            "lambda_chi": -0.03,

            "rho_xi_chi": -0.57
        }

        P_ets_t = simulate_one_gbm(**carbon_gbm_params) 
        P_gas_t = simulate_one_two_factor_schwartz_smith(**gas_schwartz_smith_params) 
        # P_gas_t = simulate_one_two_factor_schwartz_smith_ALT(**gas_schwartz_smith_params) 


        return {"P_ets_t": P_ets_t, "P_gas_t": P_gas_t}

    def calculate_cash_flows(self):
        E_t = np.array([0, 0] + [E_pmax for i in range(T_LT - 2)])
        eta_gt = 50 / 100 # % https://www.ipieca.org/resources/energy-efficiency-compendium-online/open-cycle-gas-turbines-2022#:~:text=The%20efficiency%20of%20an%20OCGT,drops%20significantly%20at%20partial%20load).
        HHV_gas = 39.24 # MJ/Sm3 https://en.wikipedia.org/wiki/Heat_of_combustion#Higher_heating_values_of_natural_gases_from_various_sources
        NG_const_t = (E_t * 3.6 * 10**9) / (eta_gt * HHV_gas) # 3.6 * 10**9 is to convert TWh to MJ - Sm^3
        electricitycosts_t = E_t * f_owf * p 
        CE_t = e * NG_const_t # carbon emissions kg CO2-eq

        P_ets_t = self.state_variables["P_ets_t"] * 11.96 # converted to NOK / tCO2-eq
        P_gas_t = self.state_variables["P_gas_t"] * 5.398175 / 100 # converted to NOK / Sm^3 

        tax_co2_t = [TAX_co2]
        Upsilon_CO2 = 2000 # NOK / tCO2
        for i in range(1, 7):
            tax_co2_t.append(tax_co2_t[i-1] + (Upsilon_CO2 - TAX_co2)/8-(i+1))
        tax_co2_t += [max(i,Upsilon_CO2)  for i in P_ets_t[7:T_LT]]

        CP_t = CE_t * 0.001 * tax_co2_t * 0.000001 + CE_t * 0.001 * P_ets_t * 0.000001 # converted to tCO2 and mnok

        revenues_t = f_owf*(NG_const_t * P_gas_t * 0.000001 + CP_t)

        CAPEX_t = 0
        OPEX_t = 0

        # TODO: electricity costs/electricity purchase expenses (from paper not below)
        cash_flows_fo_t = revenues_t - CAPEX_t - OPEX_t - electricitycosts_t

        return cash_flows_fo_t + cash_flows_wo_t


# number of simulation paths
W = 1000
monte_carlo_simulation = MonteCarlo(FieldOperatorPath, W)

monte_carlo_simulation.run_simulation()


# Assuming monte_carlo_simulation is already created and run
r = 0.08  # Example discount rate (8%)

# Visualize the results
# plot_cash_flows(monte_carlo_simulation)
plot_state_variables(monte_carlo_simulation)
plot_npv_distribution(monte_carlo_simulation, r)
# plot_npv_boxplot(monte_carlo_simulation, r)
# plot_state_variable_histograms_at_year(monte_carlo_simulation, 5)