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
with pymc.Model() as model:
    b0 =pymc.Normal('beta_0', mu=0, sd=100.)
    b1 =pymc.Normal('beta_1', mu=0, sd=100.)
    b2 =pymc.Normal('beta_2', mu=0, sd=100.)
    error =pymc.Normal('epsilon', mu=0, sd=100.)

    y_out = b0 + b1*x1 + b2*x2 
    y_var = pymc.Normal('y', mu=y_out, sd=error, observed=y)

# %%
with model:
    trace = pymc.sample(3000)

# %%
pymc.traceplot(trace)