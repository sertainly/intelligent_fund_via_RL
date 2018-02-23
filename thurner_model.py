"""
This is an implementation of the model described by Thurner, Farmer & Geanakoplos. (Leverage causes fat tails and clustered volatility, 2012)
(https://arxiv.org/abs/0908.1555)
"""

print("\n",10*"########")
print("Starting Simulation\n")

import numpy as np
import math
import time
import datetime

from utils import printProgressBar

import matplotlib.pyplot as plt
import seaborn as sns

start = time.time()

# Setting the global parameters:
# the perceived fundamental value is V
V = 1
# the total supply of the asset is N
N = 1000

class NoiseTrader:
    """
    Noise trader of the model as described on page 698
    The noise traders’ demand is defined in terms of 
    the cash value xi_t 
    """
    def __init__(self, roh, sigma, V, N):
        self.roh = roh
        self.sigma = sigma
        self.V = V
        self.N = N

    def log_spending_on_asset(self, log_xi_tm1):
        """
        Returns the log of  xi_t, 
        which is the amount of cash the noise trader
        spends on the asset. xi_t follows an autoregressive
        random process of order 1, AR(1).
        """
        log_xi_t = self.roh * log_xi_tm1 + \
                   self.sigma * np.random.randn() + \
                   (1-self.roh) * math.log(self.V*self.N)

        # xi_t = math.exp(log_xi_t)

        return log_xi_t

    def demand(self, xi_tm1, p_t):
        """
        Returns the noise trader's demand, depending on
        the current price of the asset
        """
        return self.spending_on_asset(xi_tm1) / p_t


class Fund:
    """
    The funds in our model are value investors who base 
    their demand D_h_t on a mispricing signal m_t = V - p_t.
    The perceived fundamental value V is held constant and 
    is the same for the noise traders and for all funds
    
    Funds differ only according to an aggression parameter 
    beta_h that represents how sensitive their response is 
    to the signal m.
    """
    def __init__(self, beta):
        self.beta = beta

    # TODO
    def demand(self, ...)


########################################################
# This section specifies and runs our simulation

# specifies the parameters of our noise trader (nt)
roh_nt = 0.99
sigma_nt = 0.035

# create one noise_trader
noise_trader = NoiseTrader(roh_nt, sigma_nt, V, N)

log_nt_spending = [math.log(1)]
nt_demand = []

# we set the initial price to be the mean 
# (in case of only noise traders)
prices = []

iterations = 10000

printProgressBar(0, iterations, prefix='Progress', length=50) 
for i in range(iterations):
    
    log_xi_tm1 = log_nt_spending[-1]
    log_xi_t = noise_trader.log_spending_on_asset(log_xi_tm1)
    log_nt_spending.append(log_xi_t)

    xi_t = math.exp(log_xi_t)


    # the price is determined like this if there are only nts
    p_t = xi_t / N
    prices.append(p_t)
   
    D_nt_t = xi_t / p_t 
    nt_demand.append(D_nt_t)

    printProgressBar(i + 1, iterations,
            prefix='Progress:',
            length=50)

plt.plot(prices)
plt.savefig("prices.png")
print()
print(datetime.datetime.now())
print("Done")

