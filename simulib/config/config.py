

kwargs = {   
    "CAPEX": 2000 * 10**6, # NOK
    "OPEX": 100 * 10**6, # NOK

    "r_WC": 0.07, # 0.06,
    "r_FO": 0.07, # 0.0833,

    "t_construction": 2,
    "LT_field": 6,
    "LT_turbine": 25,

    "wind_power_rating": 15, # MW
    "wind_capacity_factor": 0.525, 
    "wind_residual_value": 441 * 10**6, # Now selected to recover costs only / between 900 mNOK and 1800 mNOK

    "gas_CO2_emissions_factor": 0.00251, # tonn/Sm3 fra miljørapport på Brage
    "gas_NOx_emissions_factor": 0.00979, # kg/Sm3
    "gas_efficiency_factor": 0.3, # fra miljørapport på Brage

    "start_tax": 2.21, # kr/Sm3
    "end_tax": 6.20, # kr/Sm3
    "t_tax_ceiling": 5,
    "co2_tax_ceiling": 2650, # NOK / kg CO2 - 2400 2025-kroner justert til 2030-kroner

    "CAPEX_support": 2000 * 10**6, # NOK
    "NOx_support_rate": 500, # NOK/kg NOx
    "NOx_support_ceiling": 60 * 10**6, # NOK
}

# MWh
kwargs["wind_annual_power_production"] = (
    kwargs["wind_power_rating"] * kwargs["wind_capacity_factor"] * 24 * 365.25
)

# NG_const Sm3
kwargs["gas_burned_without_owf"] = (
    kwargs["wind_annual_power_production"] * 3.6 * 10**3 / (
        kwargs["gas_efficiency_factor"] * 40  # HHV = 40 MJ/Sm3
    )
)


gas_schwartz_smith_params = {
    "dt": 1,
    "period": kwargs["LT_field"] + 1,
    "xi_0": 4.38,
    "sigma_xi": 0.24,
    "mu_xi": -0.04,
    "kappa": 0.89,
    "chi_0": 0,
    "sigma_chi": 0.68,
    "lambda_chi": -0.03,
    "rho_xi_chi": -0.57,
}

carbon_gbm_params = {
    "mu": 0.17,
    "period": kwargs["LT_field"] + 1,
    "sigma": 0.386,
    "p_0": 97.1,
    "dt": 1,
}