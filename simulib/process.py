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
def _generate_correlated_shocks(rho_xi_chi):    
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

def simulate_one_two_factor_schwartz_smith(
        dt, period,
        xi_0, sigma_xi, mu_xi, kappa,
        chi_0, sigma_chi, lambda_chi = 0,
        rho_xi_chi = 0
):
    xi = [xi_0]
    chi = [chi_0]
    ret = []
    get_shocks = _generate_correlated_shocks(rho_xi_chi)
    for i in range(period):
        z_xi, z_t = get_shocks() # we use get_shocks function because these two are correlated rho_xi_chi
        
        xi.append(xi[i] + mu_xi * dt + sigma_xi * math.sqrt(dt) * z_xi)
        chi.append(chi[i] + (-kappa * chi[i] - lambda_chi) * dt + sigma_chi * math.sqrt(dt) * z_t)
        ret.append(math.e ** (chi[i] + xi[i]))

    return np.array(ret)