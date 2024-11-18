# import numpy as np
# import math

# rng = np.random.default_rng(seed=42)

# #######
# # GBM #
# #######
# def simulate_one_gbm(mu, sigma, p_0, dt, period):
#     ret = [p_0]
#     for i in range(1, period):
#         z = rng.normal()
#         ret.append(
#             ret[i-1] * math.e**((mu-0.5*sigma**2)*dt+sigma*z*math.sqrt(dt)) 
#         )

#     return np.array(ret)


# ######################
# # Schwartz and Smith #
# ######################
# def simulate_one_two_factor_schwartz_smith(
#         dt, period,
#         xi_0, sigma_xi, mu_xi, kappa,     # long term
#         chi_0, sigma_chi, lambda_chi = 0, # short term
#         rho_xi_chi = 0
# ):
#     xi = [xi_0]
#     chi = [chi_0]
#     ret = []
#     for i in range(period):
#         z_chi = rng.normal()
#         z_xi = rho_xi_chi * z_chi + rng.normal() * np.sqrt(1 - rho_xi_chi ** 2)

#         chi.append(
#             chi[i] * np.exp(-kappa * dt) 
#              - ((1 - np.exp(-kappa * dt)) * lambda_chi / kappa) 
#              + sigma_chi * z_chi * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
#         )
#         xi.append(xi[i] + mu_xi * dt + sigma_xi * math.sqrt(dt) * z_xi)
#         ret.append(math.e ** (chi[i] + xi[i]))

#     return np.array(ret)


# def simulate_one_two_factor_schwartz_smith_ALT(
#         dt, period,
#         xi_0, sigma_xi, mu_xi, kappa,     # long term
#         chi_0, sigma_chi, lambda_chi = 0, # short term
#         rho_xi_chi = 0
# ):
#     xi = [xi_0]
#     chi = [chi_0]
#     ret = []
#     for i in range(period):
#         z_chi = rng.normal()
#         z_xi = rho_xi_chi * z_chi + rng.normal() * np.sqrt(1 - rho_xi_chi ** 2)

#         chi.append(
#             chi[i] * np.exp(-kappa * dt) 
#              - ((1 - np.exp(-kappa * dt)) * lambda_chi / kappa) 
#              + sigma_chi * z_chi * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
#         )
#         xi.append(xi[i] + mu_xi * dt + sigma_xi * math.sqrt(dt) * z_xi)


#         # NOTE: I suspect this cannot be correct
#         # NOTE: I believe this formula is for the futures price (or expected future spot price) 
#         ret.append(math.e ** (
#                 np.exp(-kappa * i) * chi[i] + xi[i]
#                  + mu_xi * i
#                  - (1 - np.exp(-kappa * i)) * lambda_chi / kappa 
#                  + 0.5 * (
#                     (1 - np.exp(-2 * kappa * i)) * sigma_chi ** 2 / (2 * kappa) 
#                      + sigma_xi ** 2 * i 
#                      + 2 * (1 - np.exp(-kappa * i)) * rho_xi_chi * sigma_chi * sigma_xi / kappa
#                 )
#             )
#         )

#     return np.array(ret)


# # np.exp(-kappa * j) * chi[i, j] + xi[i, j] + mu_star * j \
# #     - (1 - np.exp(-kappa * j)) * lambda_chi / kappa + 0.5 * (
# #             (1 - np.exp(-2 * kappa * j)) * sigma_chi ** 2 / (2 * kappa) + sigma_xi ** 2 * j + 2 * (
# #             1 - np.exp(-kappa * j)) * rho_xi_chi * sigma_chi * sigma_xi / kappa)
import numpy as np
import math

# Set up a random number generator for reproducibility
rng = np.random.default_rng(seed=42)

#######
# GBM #
#######
def simulate_one_gbm(mu, sigma, p_0, dt, period):
    """
    Simulates one path of a geometric Brownian motion (GBM).

    Parameters:
    - mu (float): Drift of the process.
    - sigma (float): Volatility of the process.
    - p_0 (float): Initial price level.
    - dt (float): Time step size.
    - period (int): Number of time steps.

    Returns:
    - np.array: Simulated GBM price path.
    """
    z = rng.normal(0, 1, period)  # Pre-generate standard normal random values
    prices = [p_0]

    for i in range(1, period):
        prices.append(
            prices[i - 1] * math.exp((mu - 0.5 * sigma ** 2) * dt + sigma * z[i - 1] * math.sqrt(dt))
        )

    return np.array(prices)


######################
# Schwartz and Smith #
######################

def simulate_one_two_factor_schwartz_smith(
        dt, period,
        xi_0, sigma_xi, mu_xi, kappa,     # long-term factor
        chi_0, sigma_chi, lambda_chi,   # short-term factor
        rho_xi_chi
):
    xi = np.zeros(period)
    chi = np.zeros(period)
    xi[0] = xi_0
    chi[0] = chi_0

    dW_xi = rng.normal(0, np.sqrt(dt), period - 1)
    dW_chi_uncorrelated = rng.normal(0, np.sqrt(dt), period - 1)

    dW_chi = rho_xi_chi * dW_xi + np.sqrt(1 - rho_xi_chi ** 2) * dW_chi_uncorrelated

    for t in range(1, period):
        xi[t] = xi[t - 1] + mu_xi * dt + sigma_xi * dW_xi[t - 1]

        chi[t] = chi[t - 1] + kappa * (lambda_chi - chi[t - 1]) * dt + sigma_chi * dW_chi[t - 1]

    P_gas_t = np.exp(xi + chi)
    return P_gas_t

