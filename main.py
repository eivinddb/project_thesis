"""
Script for generating Monte Carlo simulations
"""
# imports
import numpy as np
import math
from simulib.montecarlo import MonteCarlo
from simulib.process import simulate_one_gbm, simulate_one_two_factor_schwartz_smith
from simulib.visualize import *

from config.config import *

ppa_price = 0

##################
# wind operator  #
##################

class WindOperatorPath(MonteCarlo.Path):

    def __init__(self) -> None:
        super().__init__()

    def simulate_state_variables(self):
        return {}

    def calculate_cash_flows(self, 
        ppa_price,
        CAPEX, OPEX, 
        t_construction, LT_field,
        wind_power_rating, wind_capacity_factor, wind_residual_value,
        CAPEX_support
    ):
        
        construction_cash_flows = np.array(
            [((CAPEX_support - CAPEX)/t_construction) for i in range(t_construction)] +
            [0 for i in range(LT_field - t_construction)] +
            [0]
        )

        operation_cash_flows = np.array(
            [0 for i in range(t_construction)] +
            [(ppa_price * wind_power_rating * wind_capacity_factor - OPEX) for i in range(LT_field - t_construction)] +
            [0]
        )

        terminal_cash_flows = np.array(
            [0 for i in range(t_construction)] +
            [0 for i in range(LT_field - t_construction)] +
            [wind_residual_value]
        )

        self.cash_flows = construction_cash_flows + operation_cash_flows + terminal_cash_flows

        return net_present_value(self.cash_flows)


##################
# field operator #
##################
class FieldOperatorPath(MonteCarlo.Path):

    def __init__(self) -> None:
        super().__init__()

    def simulate_state_variables(self):
        P_ets_t = simulate_one_gbm(**carbon_gbm_params) 
        P_gas_t = simulate_one_two_factor_schwartz_smith(**gas_schwartz_smith_params) 

        return {"P_ets_t": P_ets_t, "P_gas_t": P_gas_t}

    def calculate_cash_flows(self,
        ppa_price,
        t_construction, LT_field,
        wind_power_rating, wind_capacity_factor,
        gas_CO2_emissions_factor, gas_NOx_emissions_factor,
        start_tax, end_tax, t_tax_ceiling, co2_tax_ceiling,
        NOx_support_rate, NOx_support_ceiling,
        gas_burned_without_owf,
    ):

        construction_cash_flows = np.array(
            [0 for i in range(t_construction)] +
            [0 for i in range(LT_field - t_construction)] +
            [0]
        )

        P_ets = self.state_variables["P_ets"] * 11.96 # converted to NOK / tCO2-eq
        co2_tax_rate = np.array(
            [start_tax + i * ((end_tax - start_tax)/t_tax_ceiling) for i in range(t_tax_ceiling)] +
            [(0 if P_ets[i] > co2_tax_ceiling else (co2_tax_ceiling - P_ets[i])*gas_CO2_emissions_factor) for i in range(t_tax_ceiling, LT_field + 1)]
        )

        avoided_co2_costs = np.concatenate((
            np.array([0 for i in range(t_construction)]), 
            ((co2_tax_rate + P_ets*gas_CO2_emissions_factor)*gas_burned_without_owf)[t_construction:LT_field],
            np.array([0])
        ))

        P_gas_t = self.state_variables["P_gas_t"] * 5.398175 / 100 # converted to NOK / Sm^3 
        added_natural_gas_sales = np.concatenate((
            np.array([0 for i in range(t_construction)]), 
            (P_gas_t * gas_burned_without_owf)[t_construction:LT_field],
            np.array([0])
        ))
        
        government_funding = np.concatenate((
            np.array([0 for i in range(t_construction)]), 
            np.array([(
                (gas_NOx_emissions_factor*gas_burned_without_owf*NOx_support_rate) 
                if (gas_NOx_emissions_factor*gas_burned_without_owf*NOx_support_rate*i) < NOx_support_ceiling
                else (
                    NOx_support_ceiling - gas_NOx_emissions_factor*gas_burned_without_owf*NOx_support_rate*(i - 1)
                    if (gas_NOx_emissions_factor*gas_burned_without_owf*NOx_support_rate*(i - 1)) < NOx_support_ceiling
                    else 0
                )
            ) for i in range(LT_field - t_construction)]),
            np.array([0])
        ))

        # need to filter time
        electricity_revenue = avoided_co2_costs + added_natural_gas_sales + government_funding

        operation_cash_flows = electricity_revenue - np.array(
            [0 for i in range(t_construction)] +
            [(ppa_price * wind_power_rating * wind_capacity_factor) for i in range(LT_field - t_construction)] +
            [0]
        )

        terminal_cash_flows = np.array(
            [0 for i in range(t_construction)] +
            [0 for i in range(LT_field - t_construction)] +
            [0]
        )

        self.cash_flows = construction_cash_flows + operation_cash_flows + terminal_cash_flows

        return net_present_value(self.cash_flows)


# number of simulation paths
W = 1
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