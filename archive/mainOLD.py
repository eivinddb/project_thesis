"""
Script for generating Monte Carlo simulations
Based on Olgas epsilon paper adapted for two participants
"""
# imports
import numpy as np
import math

rng = np.random.default_rng(seed=44)

# chatgpt
def generate_correlated_shocks(rho_xi_chi):    
    # Define the covariance matrix based on the correlation
    cov_matrix = np.array([[1, rho_xi_chi], 
                           [rho_xi_chi, 1]])
    
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)

    # Inner function to generate a new pair of shocks each time it's called
    def get_shocks():
        # Generate uncorrelated standard normal variables
        uncorrelated_normals = rng.standard_normal(2)
        
        # Apply the Cholesky decomposition to introduce correlation
        correlated_normals = L @ uncorrelated_normals
        
        # Extract z_xi and z_chi from the correlated_normals
        z_xi, z_chi = correlated_normals[0], correlated_normals[1]
        return z_xi, z_chi

    return get_shocks

# Olgas exogenous parameters
CAPEX_owf = 3500 # mNOK
OPEX_owf = 200 # mNOK
TAX_co2 = 632 # NOK /tCO2
r = 8 / 100 # %
T_max = 10 # years
T_C = 2 # years
T_LT = 25 # years
f_owf = 35 / 100 # %
e = 2.364 # kg CO2-eq /Sm^3
E_pmax = 0.3 # TWh


def simulate_ets():
    # P_ets_t = P_ets_1t * math.e**((mu-0.5*sigma**2)*dt+sigma*z*math.sqrt(dt)) 
    # μ = 0.170, σ = 0.386.
    #    1    2    3    4    5    6    7    8
    # 2024 2025 2026 2027 2028 2029 2030 2031
    mu = 0.17
    sigma = 0.386
    dt = 1

    P_0 = 97.1 # EUR / tCO2-eq

    ret = [P_0]
    for i in range(1, T_LT):
        z = rng.normal()
        ret.append(
            ret[i-1] * math.e**((mu-0.5*sigma**2)*dt+sigma*z*math.sqrt(dt)) 
        )

    return np.array(ret) * 11.96 # converted to NOK / tCO2-eq


def simulate_gas():
    # two factor model
    # xi_t1 = lambda xi_t : xi_t + mu_xi * dt + sigma_xi * sqrt(dt) * z_xi
    # chi_t1 = lambda chi_t : chi_t + kappa * chi_t * dt + sigma * sqrt(dt) * z_t
    # P_t1 = lambda P_t : math.e ** (chi_t1 + xi_t1)
    xi_0 = 4.38
    sigma_xi = 0.24
    mu_xi = -0.04
    kappa = 0.89

    chi_0 = 0
    sigma_chi = 0.68
    rho_xi_chi = -0.57
    lambda_chi = -0.03

    dt = 1

    xi = [xi_0]
    chi = [chi_0]
    ret = []
    get_shocks = generate_correlated_shocks(rho_xi_chi)
    for i in range(T_LT):
        z_xi, z_t = get_shocks()
        
        xi.append(xi[i] + mu_xi * dt + sigma_xi * math.sqrt(dt) * z_xi)
        chi.append(chi[i] + (-kappa * chi[i] - lambda_chi) * dt + sigma_chi * math.sqrt(dt) * z_t)
        ret.append(math.e ** (chi[i] + xi[i]))

    return np.array(ret) * 5.398175 / 100 # converted to NOK / Sm^3 

W = 100 # number of simulation paths NOTE: unused

##################
# wind operator  #
##################
p = 0#2500 # NOK/MWh =:= mNOK/TWh
E_t = np.array([0, 0] + [E_pmax for i in range(T_LT - 2)])
revenues_t = E_t * f_owf * p 
CAPEX_t = np.array([CAPEX_owf/2, CAPEX_owf/2] + [0 for i in range(T_LT - 2)])
OPEX_t = np.array([0, 0] + [OPEX_owf for i in range(T_LT - 2)])

cash_flows_wo_t = revenues_t - CAPEX_t - OPEX_t

##################
# field operator #
##################
E_p = np.array([0, 0] + [E_pmax for i in range(T_LT - 2)])
eta_gt = 50 / 100 # % https://www.ipieca.org/resources/energy-efficiency-compendium-online/open-cycle-gas-turbines-2022#:~:text=The%20efficiency%20of%20an%20OCGT,drops%20significantly%20at%20partial%20load).
HHV_gas = 39.24 # MJ/Sm3 https://en.wikipedia.org/wiki/Heat_of_combustion#Higher_heating_values_of_natural_gases_from_various_sources
NG_const_t = (E_p * 3.6 * 10**9) / (eta_gt * HHV_gas) # 3.6 * 10**9 is to convert TWh to MJ - Sm^3
electricitycosts_t = revenues_t
CE_t = e * NG_const_t # carbon emissions kg CO2-eq

P_ets_t = simulate_ets()
P_gas_t = simulate_gas()

tax_co2_t = [TAX_co2]
Upsilon_CO2 = 2000 # NOK / tCO2
for i in range(1, 7):
    tax_co2_t.append(tax_co2_t[i-1] + (Upsilon_CO2 - TAX_co2)/8-(i+1))
tax_co2_t += [max(i,Upsilon_CO2)  for i in P_ets_t[7:T_LT]]

CP_t = CE_t * 0.001 * tax_co2_t * 0.000001 + CE_t * 0.001 * P_ets_t * 0.000001 # converted to tCO2 and mnok

revenues_t = f_owf*(NG_const_t * P_gas_t * 0.000001 + CP_t)

CAPEX_t = 0
OPEX_t = 0

# TODO: electricity costs/electricity purchase expenses residual value
cash_flows_fo_t = revenues_t - CAPEX_t - OPEX_t

## print
print(cash_flows_fo_t)
print(cash_flows_wo_t)
print(sum(cash_flows_wo_t + cash_flows_fo_t))
def net_present_value(cashflows):
    """
    Calculate the Net Present Value (NPV) for a series of cash flows.
    
    Parameters:
    cashflows (ndarray): Array of cash flows, where index represents time t.
    r (float): Discount rate.
    
    Returns:
    float: Net Present Cashflow (NPC).
    """
    t = np.arange(T_LT)  # Time periods (0, 1, 2, ..., n-1)
    discounted_cashflows = cashflows / (1 + r) ** t  # Discount each cash flow
    npv = np.sum(discounted_cashflows)  # Sum of discounted cash flows
    return npv


print(
    net_present_value(cash_flows_wo_t + cash_flows_fo_t)
)