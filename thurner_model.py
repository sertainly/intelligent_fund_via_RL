"""
This is an implementation of the model described by Thurner, Farmer & Geanakoplos. (Leverage causes fat tails and clustered volatility, 2012)
(https://arxiv.org/abs/0908.1555)

Inspiration comes from Luzius Meisser's implementation of the same model (his is for educational purposes, so much better explained)
https://github.com/kronrod/sfi-complexity-mooc/blob/master/notebooks/leverage.ipynb
"""

print("\n",10*"########")
print("Starting Simulation\n")

import numpy as np
import math
import time
import datetime

from utils import printProgressBar
from testing import calculateLogReturns, calculateAutocorrelation

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# PLOTTING
outpath = ("./figures/")

start = time.time()

# Setting the global parameters:
# the perceived fundamental value is V
V = 1
# the total supply of the asset is N
N = 1000

# alpha is used to adjust the market price
alpha = 0.1

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

    def cash_spending(self, xi_tm1):
        """
        IN: xi_tm1, the cash spent on asset in prev. period 
        OUT:xi_t, which is the amount of cash the noise trader
            spends on the asset in this period t.
            xi_t follows an autoregressive random process
            of order 1, AR(1).
        """
        log_xi_tm1 = math.log(xi_tm1)
        
        log_xi_t = self.roh * log_xi_tm1 + \
                   self.sigma * np.random.randn() + \
                   (1-self.roh) * math.log(self.V*self.N)

        xi_t = math.exp(log_xi_t)

        return xi_t

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
    
    initial_wealth = 2
    lambda_max = 20
    
    def __init__(self, beta):
        self.beta = beta
        self.cash = self.initial_wealth
        self.shares = 0 
        self.activation_delay = 0
        
    def check_and_make_bankrupt(self, p_t):
        """
        Checks whether a fund goes bankrupt
        If so, reset its shares and wealth, set an
        activation_delay, after which the fund reinitiates
        (if the fund is already bankrupt,
        the activation_delay is decreased)
        """
        # can only go bankrupt if active
        if self.is_active():
            # if wealth drops below 10% of initial_wealth
            if self.get_wealth(p_t) <= self.initial_wealth*0.1:
                # "make" fund bankrupt
                self.shares = 0
                self.cash = self.initial_wealth
                self.activation_delay = 100
                return True
            
            # if fund in not bankrupt, return False
            else: return False

        # if fund is already bankrupt, reduce activation_delay
        else:
            self.activation_delay -= 1
            return False

    def get_wealth(self, p_t):
        """
        TODO
        """
        return self.shares * p_t + self.cash

    #!!! for learning behaviour see LeBaron2012 !!!

    def get_demand(self, p_t):
        """
        Oh look, a docstring
        """ 
        # the mispricing signal m_t is the difference
        # between the fundamental value V and the price p_t
        m_t = V - p_t
        m_critical = self.lambda_max / self.beta        
        
        # if the mispricing signal m_t is positive and 
        # m_critical is not reached, yet
        if 0 < m_t and m_t < m_critical:
            return self.beta * m_t * self.get_wealth(p_t) / p_t 
        # if m_critical is reached, fund leverages 
        # to the maximum 
        elif m_t >= m_critical:
            return self.lambda_max * self.get_wealth(p_t) / p_t

        # if m_t < 0, ie the mispricing signal, is negative,
        # demand is zero
        else:
            return 0

    def is_active(self):
        return self.activation_delay == 0

    def process_inflows(self, oldprice, newprice):
        pass #used later

    def update_fund_holdings(self, p_t):
        wealth = self.get_wealth(p_t)
        self.shares = self.get_demand(p_t)
        self.cash = wealth - self.shares * p_t
########################################################
# Market clearing mechanism

# see LeBaron 2006 for possible price determination machanisms
# Thurner et al. numerically clear the market each period
# two options are: Day & Huang 1990, or Farmer et al. 2005
def update_price(p_tm1, total_demand):
    """from LeBaron 2006"""
    return p_tm1 + alpha * (total_demand - N)


########################################################
# This section initializes our agents and runs the simulation

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
    beta_of_fund = (h+1)*5
    funds.append(Fund(beta_of_fund))

# SIMULATION
prices = []

#TODO
iterations = 10

printProgressBar(0, iterations, prefix='Progress', length=50) 

for i in range(iterations):

    # Noise trader demand 
    xi_tm1 = nt_spending[-1]
    xi_t = noise_trader.cash_spending(xi_tm1)
    nt_spending.append(xi_t)

    # the price is determined like this if there are only nts
    p_t = xi_t / N
    prices.append(p_t)

    printProgressBar(i + 1, iterations,
            prefix='Progress:',
            length=50)

log_prices = np.log(np.array(prices))
print("\n")
print("Average Log Price: {}".format(sum(log_prices)/len(log_prices)))

# Test the fund demand function
testFund = Fund(50)

prices = []
demand = []
for i in range(-99, 100):
    price = V + i / 100
    investment = testFund.get_demand(price) * price
    prices.append(price)
    demand.append(investment)
    
plt.xlabel('Price')
plt.ylabel('Investment')
plt.axis([0.0, 1.5, 0, 50])
plt.plot(prices, demand)
plt.savefig("{}fund_demand.png".format(outpath))


# From Meisser's Version:
returns = calculateLogReturns(prices)

# This should output a slightly negative value. If not, you have been unlucky and should run it again. :)
print("Autocorrelation: {}".format(calculateAutocorrelation(returns, 1)))


plt.figure(figsize=(16,3))
plt.plot(prices)
plt.savefig("{}prices.png".format(outpath))
print()
print(datetime.datetime.now())
print("Done")

