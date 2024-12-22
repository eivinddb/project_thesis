

kwargs = {   
    "CAPEX_WC": 1200 * 10**6, # NOK
    "CAPEX_FO": 180 * 10**6, # NOK
    "OPEX": 24 * 10**6, # NOK 2% of CAPEX

    "r": 0.07,#0.06, # 0.06,
    # "r_FO": 0.07,#0.0833, # 0.0833,

    "t_construction": 2,
    "LT_field": 6,
    "LT_turbine": 25,

    "wind_power_rating": 15, # MW
    "wind_capacity_factor": 0.525, 
    "wind_residual_ppa": 1200,

    "gas_CO2_emissions_factor": 0.00251, # tonn/Sm3 fra miljørapport på Brage
    "gas_NOx_emissions_factor": 0.00979, # kg/Sm3
    "gas_efficiency_factor": 0.3, # fra miljørapport på Brage

    "start_tax": 2.21, # kr/Sm3
    "end_tax": 6.65, # kr/Sm3
    "t_tax_ceiling": 5,
    "co2_tax_ceiling": 2650, # NOK / kg CO2 - 2400 2025-kroner justert til 2030-kroner

    "CAPEX_support": 400 * 10**6, # NOK based on rates for Hywind and GoliatVind
    "NOx_support_rate": 500, # NOK/kg NOx
    "NOx_support_ceiling": 60 * 10**6, # NOK

    "gas_price_conversion_factor": 5.360249, # (thrm/Sm3)*(NOK/GBP)
    "carbon_price_conversion_factor": 11.96, # (NOK/EUR)

    "carbon_price_ceiling": 250
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
    "xi_0": 4.38 - 4.605170186 + 0.5730061863,
    "sigma_xi": 0.24,
    "mu_xi": -0.04,
    "kappa": 0.89,
    "chi_0": 0,
    "sigma_chi": 0.68,
    "lambda_chi": -0.03,
    "rho_xi_chi": -0.57,
}


carbon_gbm_params = {
    "mu": 0.178,#0.0616,
    "period": kwargs["LT_field"] + 1,
    "sigma": 0.472,#0.4807,
    "p_0": 68.650,	# https://www.ice.com/products/197/EUA-Futures/data?marketId=7937862
    "dt": 1,
}