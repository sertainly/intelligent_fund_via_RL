"""
This is an implementation of the model described by Thurner, Farmer & Geanakoplos. (Leverage causes fat tails and clustered volatility, 2012)
(https://arxiv.org/abs/0908.1555)

Inspiration comes from Luzius Meisser's jupyter notebook of the same model (his is for educational purposes, so much better explained)
https://github.com/kronrod/sfi-complexity-mooc/blob/master/notebooks/leverage.ipynb
"""

import scipy
import numpy as np
import math

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
        return max(0, self.shares * p_t + self.cash)

    #!!! for learning behaviour see LeBaron2012 !!!

    def get_demand(self, p_t):
        """
        Oh look, a docstring
        """ 
        if self.is_active():
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
            else: return 0
        else: return 0

    def is_active(self):
        return self.activation_delay == 0

    def process_inflows(self, oldprice, newprice):
        pass #used later

    def update_holdings(self, p_t):
        if self.is_active():
            wealth = self.get_wealth(p_t)
            self.shares = self.get_demand(p_t)
            self.cash = wealth - self.shares * p_t

        
# DynamicFund extends Fund by adding inflow/outflow dynamics (by Meisser)
class DynamicFund(Fund):
    
    a = 0.2
    benchmark_performance = 0.005 # r^b
    sensitivity = 0.10 # b, original paper uses 0.15, but 0.10 looks more interesting to me
        
    def __init__(self, aggressiveness):
        super(DynamicFund, self).__init__(aggressiveness)
        self.performance = 0.0
        self.previous_wealth = self.initial_wealth
        self.previous_investment = 0.0
        self.ret = 0.0
        
    def update_performance(self, oldprice, newprice, wealth):
        self.ret = (newprice/oldprice - 1)*self.previous_investment/self.previous_wealth
        self.performance = (1-self.a) * self.performance + self.a * self.ret # equation 5
        # remember values for next round
        self.previous_investment = self.shares * newprice
        self.previous_wealth = wealth
        return self.performance
    
    def process_inflows(self, oldprice, newprice):
        if self.is_active():
            wealth = self.get_wealth(newprice)
            perf = self.update_performance(oldprice, newprice, wealth)
            inflow = self.sensitivity*(perf - self.benchmark_performance)*wealth
            self.cash += max(inflow, -wealth)        
        
########################################################
# Market clearing mechanism

# see LeBaron 2006 for possible price determination machanisms
# Thurner et al. numerically clear the market each period
# two options are: Day & Huang 1990, or Farmer et al. 2005
def update_price(p_tm1, total_demand):
    """from LeBaron 2006"""
    return p_tm1 + alpha * (total_demand - N)

#from Meisser:
minPrice = 0.01
maxPrice = 5

# Rearranged equation 4
def calculate_excess_demand(xi_t, funds, p_t):
    demand = xi_t / p_t
    for f in funds:
        demand += f.get_demand(p_t)
    return demand - N
        
def find_equilibrium(xi_t, funds):
    # The scipy solver wants an univariate function,
    # so we create a temporary demand function 
    # that only depends on p_t, with the other two
    # parameters staying constant
    
    current_excess_demand = \
        lambda  p_t : calculate_excess_demand(xi_t, funds, p_t)
    
    return scipy.optimize.brentq(current_excess_demand,
                                 minPrice,
                                 maxPrice)
