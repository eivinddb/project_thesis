
import numpy as np

def net_present_value(cashflows, r):
    """
    Calculate the Net Present Value (NPV) for a series of cash flows.
    
    Parameters:
    cashflows (ndarray): Array of cash flows, where index represents time t.
    r (float): Discount rate (as a decimal, e.g., 0.08 for 8%).
    
    Returns:
    float: Net Present Value (NPV).
    """
    t = np.arange(len(cashflows))  # Time periods (0, 1, 2, ..., n-1)
    discounted_cashflows = cashflows / (1 + r) ** t  # Discount each cash flow
    npv = np.sum(discounted_cashflows)  # Sum of discounted cash flows
    return npv