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

from thurner_model import Fund, NoiseTrader

# 1. Initialize Agents

# NOISE TRADER 
roh_nt = 0.99
sigma_nt = 0.035
noise_trader = NoiseTrader(roh_nt, sigma_nt, V, N)

nt_spending = [N*V]

# FUNDS
funds = []
number_of_funds = 10

for h in range(number_of_funds):
    # betas range from 5 to 50 (5,10,15,...,50)
    beta_h = (h+1)*5
    funds.append(Fund(beta_h))

# 2. SIMULATION
prices = []


#TODO
iterations = 10

printProgressBar(0, iterations, prefix='Progress', length=50) 

for i in range(iterations):

    # Noise trader demand 
    xi_tm1 = nt_spending[-1]
    xi_t = noise_trader.cash_spending(xi_tm1)
    nt_spending.append(xi_t)

    # Fund demand
    
    p_t = xi_t / N
    prices.append(p_t)

    printProgressBar(i + 1, iterations,
            prefix='Progress:',
            length=50)

log_prices = np.log(np.array(prices))
print("\n")
print("Average Log Price: {}".format(sum(log_prices)/len(log_prices)))



