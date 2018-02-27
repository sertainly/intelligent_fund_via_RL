# Verify that in the long run, returns are slightly negatively autocorrelated due to mean-reversion.
# This is the opposite of having momentum (which can be often ).
# With mean-reversion, the price is a little more likely to go down at time t+1 if it went up at time t.

import numpy as np
import math

def calculateLogReturn(x, y):
    return math.log(x/y)

def calculateLogReturns(prices):
    length = len(prices)
    return list(map(calculateLogReturn, prices[0:length-1], prices[1:length]))

def calculateAutocorrelation(returns, lag):
    length = len(returns)

    return np.corrcoef(returns[0:length-lag], returns[lag:length])[0][1]
