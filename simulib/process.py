import numpy as np
import math

rng = np.random.default_rng(seed=42)

#######
# GBM #
#######
def simulate_one_gbm(mu, sigma, p_0, dt, period):
    ret = [p_0]
    for i in range(1, period):
        z = rng.normal()
        ret.append(
            ret[i-1] * math.e**((mu-0.5*sigma**2)*dt+sigma*z*math.sqrt(dt)) 
        )

    return np.array(ret)


######################
# Schwartz and Smith #
######################
def simulate_one_two_factor_schwartz_smith(
        dt, period,
        xi_0, sigma_xi, mu_xi, kappa,     # long term
        chi_0, sigma_chi, lambda_chi = 0, # short term
        rho_xi_chi = 0
):
    xi = [xi_0]
    chi = [chi_0]
    ret = []
    for i in range(period):
        z_chi = rng.normal()
        z_xi = rho_xi_chi * z_chi + rng.normal() * np.sqrt(1 - rho_xi_chi ** 2)

        chi.append(
            chi[i] * np.exp(-kappa * dt) 
             - ((1 - np.exp(-kappa * dt)) * lambda_chi / kappa) 
             + sigma_chi * z_chi * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
        )
        xi.append(xi[i] + mu_xi * dt + sigma_xi * math.sqrt(dt) * z_xi)
        ret.append(math.e ** (chi[i] + xi[i]))

    return np.array(ret)


def simulate_one_two_factor_schwartz_smith_ALT(
        dt, period,
        xi_0, sigma_xi, mu_xi, kappa,     # long term
        chi_0, sigma_chi, lambda_chi = 0, # short term
        rho_xi_chi = 0
):
    xi = [xi_0]
    chi = [chi_0]
    ret = []
    for i in range(period):
        z_chi = rng.normal()
        z_xi = rho_xi_chi * z_chi + rng.normal() * np.sqrt(1 - rho_xi_chi ** 2)

        chi.append(
            chi[i] * np.exp(-kappa * dt) 
             - ((1 - np.exp(-kappa * dt)) * lambda_chi / kappa) 
             + sigma_chi * z_chi * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
        )
        xi.append(xi[i] + mu_xi * dt + sigma_xi * math.sqrt(dt) * z_xi)


        # NOTE: I suspect this cannot be correct
        # NOTE: I believe this formula is for the futures price (or expected future spot price) 
        ret.append(math.e ** (
                np.exp(-kappa * i) * chi[i] + xi[i]
                 + mu_xi * i
                 - (1 - np.exp(-kappa * i)) * lambda_chi / kappa 
                 + 0.5 * (
                    (1 - np.exp(-2 * kappa * i)) * sigma_chi ** 2 / (2 * kappa) 
                     + sigma_xi ** 2 * i 
                     + 2 * (1 - np.exp(-kappa * i)) * rho_xi_chi * sigma_chi * sigma_xi / kappa
                )
            )
        )

    return np.array(ret)


# np.exp(-kappa * j) * chi[i, j] + xi[i, j] + mu_star * j \
#     - (1 - np.exp(-kappa * j)) * lambda_chi / kappa + 0.5 * (
#             (1 - np.exp(-2 * kappa * j)) * sigma_chi ** 2 / (2 * kappa) + sigma_xi ** 2 * j + 2 * (
#             1 - np.exp(-kappa * j)) * rho_xi_chi * sigma_chi * sigma_xi / kappa)