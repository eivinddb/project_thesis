import numpy as np

def priceSS(E0, mu, sigmaE, X0, kappa, sigmaX, lambdaX, rho, T, Nstep, Ntrial):
    """
    Generate short and long term components, Chi and Xi
 
    @return: long, short
    """
    dt = T / Nstep
    short = np.zeros((Ntrial, Nstep + 1))
    long = np.zeros((Ntrial, Nstep + 1))
    long[:, 0] = E0
    short[:, 0] = X0

    for j in range(0, Nstep-1):
        epsilon = np.random.randn(1, Ntrial)
        
        short[:, j+1] = short[:, j] * np.exp(-kappa * dt) - (
                (1 - np.exp(-kappa * dt)) * lambdaX / kappa) + sigmaX * epsilon * np.sqrt(
            (1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
        
        epsilonE = rho * epsilon + np.sqrt(1 - rho ** 2) * np.random.randn(1, Ntrial)
        long [:, j+1] = long[:, j] + mu * dt + sigmaE * np.sqrt(dt) * epsilonE
    return long, short
 
def futures(long, mu_star, sigmaE, short, kappa, sigmaX, lambdaX, rho, t, Ntrial):
    """
    Generate future prices (%/bbl)
 
    @return: futures[path,time]
    """
    F = np.zeros((Ntrial, t))
    lnF = np.zeros((Ntrial, t))
    for i in range(Ntrial):
        for j in range(t):
            lnF[i, j] = np.exp(-kappa * j) * short[i, j] + long[i, j] + mu_star * j \
                        - (1 - np.exp(-kappa * j)) * lambdaX / kappa + 0.5 * (
                                (1 - np.exp(-2 * kappa * j)) * sigmaX ** 2 / (2 * kappa) + sigmaE ** 2 * j + 2 * (
                                1 - np.exp(-kappa * j)) * rho * sigmaX * sigmaE / kappa)
            F[i, j] = np.exp(lnF[i, j])
    # print(F[:t])
 
    return F




def simulate_one_two_factor_schwartz_smith(xi_0, mu_xi, sigma_xi, chi_0, kappa, sigma_chi, lambda_chi, rho_xi_chi, T, period, Ntrial):
    """
    Generate long-term and short-term components, Xi and Chi.
    
    @return: xi, chi
    """
    dt = T / period
    chi = np.zeros((Ntrial, period + 1))
    xi = np.zeros((Ntrial, period + 1))
    xi[:, 0] = xi_0
    chi[:, 0] = chi_0
    
    for j in range(0, period - 1):
        z_chi = np.random.randn(1, Ntrial)

        chi[:, j + 1] = chi[:, j] * np.exp(-kappa * dt) - (
                (1 - np.exp(-kappa * dt)) * lambda_chi / kappa) + sigma_chi * z_chi * np.sqrt(
            (1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
        
        z_xi = rho_xi_chi * z_chi + np.sqrt(1 - rho_xi_chi ** 2) * np.random.randn(1, Ntrial) # NOTE: this is usable

        xi[:, j + 1] = xi[:, j] + mu_xi * dt + sigma_xi * np.sqrt(dt) * z_xi  # NOTE: this is the same
        
    return xi, chi


def futures(xi, mu_star, sigma_xi, chi, kappa, sigma_chi, lambda_chi, rho_xi_chi, t, Ntrial):
    """
    Generate future prices (%/bbl).
    
    @return: futures[path, time]
    """
    F = np.zeros((Ntrial, t))
    lnF = np.zeros((Ntrial, t))
    
    for i in range(Ntrial):
        for j in range(t):
            lnF[i, j] = np.exp(-kappa * j) * chi[i, j] + xi[i, j] + mu_star * j \
                        - (1 - np.exp(-kappa * j)) * lambda_chi / kappa + 0.5 * (
                                (1 - np.exp(-2 * kappa * j)) * sigma_chi ** 2 / (2 * kappa) + sigma_xi ** 2 * j + 2 * (
                                1 - np.exp(-kappa * j)) * rho_xi_chi * sigma_chi * sigma_xi / kappa)
            F[i, j] = np.exp(lnF[i, j])
    
    return F