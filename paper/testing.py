# Verify that in the long run, returns are slightly negatively autocorrelated due to mean-reversion.
# This is the opposite of having momentum (which can be often ).
# With mean-reversion, the price is a little more likely to go down at time t+1 if it went up at time t.


from thurner_model import NoiseTrader, Fund

import numpy as np
import math

import matplotlib.pyplot as plt

outpath = "./figures/"

# Setting the global parameters:
# the perceived fundamental value is V
V = 1
# the total supply of the asset is N
N = 1000

# alpha is used to adjust the market price
alpha = 0.1

#######################################################

def calculateLogReturn(x, y):
    return math.log(x/y)

def calculateLogReturns(prices):
    length = len(prices)
    return list(map(calculateLogReturn, prices[0:length-1], prices[1:length]))

def calculateAutocorrelation(returns, lag):
    length = len(returns)

    return np.corrcoef(returns[0:length-lag], returns[lag:length])[0][1]


########################################################
# Test the fund demand function
def test_fund_demand():
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
    plt.close()

########################################################
# test the market clearing mechanism
def test_market_clearing():
    funds = []
    for shares in range(1, 11):
        f = Fund(10)
        f.shares = shares * 2
        funds.append(f)
    
    nt_spending = []
    prices = [2]
    
    funds_demand_all_periods = []
    
    for inv in range(1, 140):
        
        p_t = prices[-1]
        
        xi_t = inv * 10
        nt_spending.append(xi_t)
        nt_demand = xi_t / p_t
    
        funds_total_demand_t = []
    
        for f in funds:
            fund_demand = f.get_demand(p_t)
            funds_total_demand_t.append(fund_demand)
    
        funds_demand_all_periods.append(funds_total_demand_t)
        funds_total_demand_t = sum(funds_total_demand_t)
    
        total_demand = nt_demand + funds_total_demand_t
    
        p_t = update_price(p_t, total_demand)
        prices.append(p_t)
    
    demand_per_period = []
    for demand in funds_demand_all_periods:
        demand_per_period.append(sum(demand))
    
    plt.plot(demand_per_period)
    plt.savefig("{}demand_per_period".format(outpath))
    plt.close()
    
    prices = prices[1:]
    print(len(prices))
    plt.xlabel('Noise Trader Spending xi_t')
    plt.ylabel('Price')
    plt.plot(nt_spending, prices)
    plt.savefig("{}nt_spending_market_mechanism".format(outpath))



######################## Provoke Margin Call
def provoke_margin_call():
    fund = Fund(30)
    
    days = []
    prices = []
    wealth = []
    investment = []
    for i in range(120):
        price = 1.4 - i/100
        fund.update_fund_holdings(price)
        days.append(i)
        prices.append(price)
        wealth.append(fund.get_wealth(price))
        investment.append(fund.shares * price) 

    plt.plot(days, prices, 'b', label='price')
    plt.plot(days, investment, 'g', label='investment')
    plt.plot(days, wealth, 'r', label='wealth')
    plt.xlabel('day')
    plt.ylabel('amount')
    plt.legend(loc='upper left')
    plt.savefig("{}margin_call.png".format(outpath))
    plt.close()

#########################################################

#From Meisser's Version:
#returns = calculateLogReturns(prices)

# This should output a slightly negative value.
# If not, you have been unlucky and should run it again. :)
#print("Autocorrelation: {}".format(
#                       calculateAutocorrelation(returns, 1)))

#plt.figure(figsize=(16,3))
#plt.plot(prices)
#plt.savefig("{}prices.png".format(outpath))
#
#print("\nDone")

