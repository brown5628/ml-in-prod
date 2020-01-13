#%%
import numpy as np 
population = np.random.uniform(5,15, size =1000)


# %%
sample = np.random.choice(population, size =30, replace=False)

# %%
population.mean() 

# %%
sample.mean()

# %%
lower_range = sample.mean() - 1.96 * sample.std(ddof=1) / np.sqrt(len(sample))

upper_range = sample.mean() + 1.96 * sample.std(ddof=1) / np.sqrt(len(sample))

# %%
