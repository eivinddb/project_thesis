"""
Script for generating Monte Carlo simulations
"""
# imports
import numpy as np
import math

from .montecarlo import MonteCarlo
from .process import simulate_one_gbm, simulate_one_two_factor_schwartz_smith
from .utils import *
from .config.config import *


class WindContractorPath(MonteCarlo.Path):

    def __init__(self) -> None:
        super().__init__()

    def simulate_state_variables(self):
        return {}

    def calculate_cash_flows(self, 
        ppa_price,
        CAPEX_WC, OPEX,
        r_WC, 
        t_construction, LT_field,
        wind_annual_power_production, wind_residual_value,
        CAPEX_support,
        **kwargs
    ):
        
        self.CAPEX_support_t = np.array(
            [(CAPEX_support/t_construction) for i in range(t_construction)] +
            [0 for i in range(LT_field - t_construction)] +
            [0]
        )

        self.CAPEX_t = np.array(
            [(-CAPEX_WC/t_construction) for i in range(t_construction)] +
            [0 for i in range(LT_field - t_construction)] +
            [0]
        )

        self.electricity_revenue_t = np.array(
            [0 for i in range(t_construction)] +
            [(ppa_price * wind_annual_power_production) for i in range(LT_field - t_construction)] +
            [0]
        )

        self.OPEX_t = np.array(
            [0 for i in range(t_construction)] +
            [(-OPEX) for i in range(LT_field - t_construction)] +
            [0]
        )

        self.wind_residual_value_t = np.array(
            [0 for i in range(t_construction)] +
            [0 for i in range(LT_field - t_construction)] +
            [wind_residual_value]
        )

        self.cash_flows = (
            self.CAPEX_support_t + self.CAPEX_t 
            + self.OPEX_t 
            + self.electricity_revenue_t
            + self.wind_residual_value_t
        )
        
        # unused
        self.tax = self.cash_flows * 0.22

        return net_present_value(self.cash_flows, r_WC)


class FieldOperatorPath(MonteCarlo.Path):

    def __init__(self, carbon_gbm_params = carbon_gbm_params, gas_schwartz_smith_params = gas_schwartz_smith_params) -> None:
        self.carbon_gbm_params = carbon_gbm_params
        self.gas_schwartz_smith_params = gas_schwartz_smith_params
        super().__init__()

    def simulate_state_variables(self):
        P_ets_t = simulate_one_gbm(**self.carbon_gbm_params) 
        P_gas_t = simulate_one_two_factor_schwartz_smith(**self.gas_schwartz_smith_params) 

        return {"P_ets": P_ets_t, "P_gas": P_gas_t}

    def calculate_cash_flows(self,
        ppa_price,
        CAPEX_FO,
        r_FO,
        t_construction, LT_field,
        wind_annual_power_production,
        gas_CO2_emissions_factor, gas_NOx_emissions_factor,
        start_tax, end_tax, t_tax_ceiling, co2_tax_ceiling,
        NOx_support_rate, NOx_support_ceiling,
        gas_burned_without_owf,
        **kwargs
    ):

        # Capex
        CAPEX_FO_t = np.array(
            [-CAPEX_FO/t_construction for i in range(t_construction)] +
            [0 for i in range(LT_field - t_construction)] +
            [0]
        )

        # Carbon costs - cap at 250 EUR
        P_ets = np.minimum(self.state_variables["P_ets"], 250) * 11.96 # converted to NOK / tCO2-eq
        
        co2_tax_rate = np.array(
            [start_tax + i * ((end_tax - start_tax)/t_tax_ceiling) for i in range(t_tax_ceiling)] +
            [(0 if P_ets[i] > co2_tax_ceiling else (co2_tax_ceiling - P_ets[i])*gas_CO2_emissions_factor) for i in range(t_tax_ceiling, LT_field + 1)]
        )

        avoided_co2_tax_t = np.concatenate((
            np.array([0 for i in range(t_construction)]), 
            ((co2_tax_rate)*gas_burned_without_owf)[t_construction:LT_field],
            np.array([0])
        ))

        avoided_co2_allowance_t = np.concatenate((
            np.array([0 for i in range(t_construction)]), 
            ((P_ets*gas_CO2_emissions_factor)*gas_burned_without_owf)[t_construction:LT_field],
            np.array([0])
        ))

        # Added gas sales https://ngc.equinor.com
        P_gas_t = self.state_variables["P_gas"] * 5.360249 # converted to NOK / Sm^3 
        
        added_natural_gas_sales_t = np.concatenate((
            np.array([0 for i in range(t_construction)]), 
            (P_gas_t * gas_burned_without_owf)[t_construction:LT_field],
            np.array([0])
        ))
        
        # NOx fund
        government_funding_t = np.concatenate((
            np.array([0 for i in range(t_construction)]), 
            np.array([(
                (gas_NOx_emissions_factor*gas_burned_without_owf*NOx_support_rate) 
                if (gas_NOx_emissions_factor*gas_burned_without_owf*NOx_support_rate*(i+1)) < NOx_support_ceiling
                else (
                    NOx_support_ceiling - gas_NOx_emissions_factor*gas_burned_without_owf*NOx_support_rate*(i)
                    if (gas_NOx_emissions_factor*gas_burned_without_owf*NOx_support_rate*(i)) < NOx_support_ceiling
                    else 0
                )
            ) for i in range(LT_field - t_construction)]),
            np.array([0])
        ))

        # Electricity costs (payments to wind contractor)
        electricity_costs = - np.array(
            [0 for i in range(t_construction)] +
            [(ppa_price * wind_annual_power_production) for i in range(LT_field - t_construction)] +
            [0]
        )
         
        self.cash_flows = (
            CAPEX_FO_t 
            + avoided_co2_tax_t + avoided_co2_allowance_t 
            + added_natural_gas_sales_t
            + government_funding_t
            + electricity_costs
        )

        # tax here, unused
        self.tax = self.cash_flows * 0.78

        return net_present_value(self.cash_flows, r_FO)


def get_discounted_power_production(
        wind_annual_power_production, 
        t_construction, LT_field,
        discount_rate,
        **kwargs):
    power_production_flows = np.array(
        [0 for i in range(t_construction)] +
        [wind_annual_power_production for i in range(LT_field - t_construction)]
    )
    return net_present_value(power_production_flows, discount_rate)


