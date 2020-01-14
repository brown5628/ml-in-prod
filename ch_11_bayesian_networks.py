# %%
import pandas as pd 
import numpy as np 
import pymc3 as pymc 

n = 10000
beta0 = -1
beta1 = 1. 
beta2 = 2. 

x1 = np.random.normal(size=n)
x2 = np.random.normal(size=n)
y = np.random.normal(beta1 * x1 + beta2 * x2 + beta0)


# %%
