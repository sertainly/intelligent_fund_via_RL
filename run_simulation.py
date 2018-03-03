"""
This module runs the simulation.
"""
print("\n",10*"########")
print("Starting Simulation\n")

import time
import datetime

from utils import printProgressBar

import scipy
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# PLOTTING
outpath = ("./figures/")

from thurner_model import Fund, NoiseTrader, find_equilibrium

# 1. Initialize Agents

def simulate(iterations):
    # the perceived fundamental value is V
    V = 1
    # the total supply of the asset is N
    N = 1000

    # 1. Initialize agents and parameters

    # Initialize our noise trader 
    roh_nt = 0.99
    sigma_nt = 0.035
    noise_trader = NoiseTrader(roh_nt, sigma_nt, V, N)
    
    nt_spending = [N*V]
    
    # Initialize our funds 
    funds = []
    number_of_funds = 5 
    
    for h in range(number_of_funds):
        # betas range from 5 to 50 (5,10,15,...,50)
        beta_h = (h+1)*5
        funds.append(Fund(beta_h))
    
    # 2. SIMULATION
    
    initial_price = 1
    prices = [initial_price]
    total_fund_wealth = []
    
    printProgressBar(0, iterations,
                    prefix='Progress', length=50) 
    
    for i in range(iterations):
    
        p_tm1 = prices[-1]
        # Noise trader spending 
        xi_tm1 = nt_spending[-1]
        xi_t = noise_trader.cash_spending(xi_tm1)
        nt_spending.append(xi_t)
    
        p_t = find_equilibrium(xi_t, funds)
    
        # Fund demand
        funds_wealth_t = [] 
    
        for fund in funds:
            fund.update_holdings(p_t)
            fund.check_and_make_bankrupt(p_t)
            fund.process_inflows(p_tm1, p_t)
            wealth_fund_t = fund.get_wealth(p_t)
            funds_wealth_t.append(wealth_fund_t)
   
        total_fund_wealth.append(funds_wealth_t)
    
        prices.append(p_t)
    
        printProgressBar(i + 1, iterations,
                prefix='Progress:',
                length=50)

    prices = prices[1:]

    return {'iterations':iterations,
            'prices':prices,
            'wealth':total_fund_wealth,
            'num_funds': number_of_funds}


def plot_prices(result):
    plt.figure(figsize=(16,4))
    plt.plot(result['prices'], 'b',
             label='including funds')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.savefig("{}simulated_prices_{}_funds.png".format(outpath, result['num_funds']))
    plt.close()

# Run the simulation
result = simulate(iterations=10000)

# Plot prices
plot_prices(result)

# Plot the wealth of all funds
plt.plot(result['wealth'])
plt.savefig("{}simulated_wealth_{}_funds.png".format(outpath, result['num_funds']))
plt.close()




#############################################
#log_prices = np.log(np.array(prices))
#print("\n")
#print("Average Log Price: {}".format(sum(log_prices)/len(log_prices)))



