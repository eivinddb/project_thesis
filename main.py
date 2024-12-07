
from scipy.optimize import brentq
from simulib.cash_flow_models import *


a = get_discounted_power_production(
        wind_annual_power_production = 69032.25,
        t_construction = 2, LT_field = 26,
        discount_rate = 0.06
    ),
b = net_present_value([200, 200] + 25*[100], 0.06)

print(b/a*1000*1000)


# zero_level = brentq(objective, CAPEX_low, CAPEX_high)


# ## Main
# ppa_price = 500 # kr/MWh basert på utsira Nord "high"-case
# W = 5 # number of simulation paths

# print()
# wc_simulation = MonteCarlo(WindContractorPath, 1)
# fo_simulation = MonteCarlo(FieldOperatorPath, W)

# print("\nNet Present Energy (MWh)", get_discounted_power_production(discount_rate=0.07, **kwargs))

# npv_wc = wc_simulation.calculate_all_cash_flows(ppa_price = ppa_price, **kwargs)
# npvs_fo = fo_simulation.calculate_all_cash_flows(ppa_price = ppa_price, **kwargs)

# print("\nWind Contractor")
# print_currency_array("Cash flow:\n", wc_simulation.paths[0].cash_flows)
# print_currency_array("NPV:", npv_wc)

# print("\nField Operator")
# print("Chash Flows:")
# for i in fo_simulation.paths:
#     print_currency_array("", i.cash_flows)
# print_currency_array("NPV:", npvs_fo)

# print_currency_array("\nNet Project NPV", np.mean(npvs_fo)+npv_wc)
