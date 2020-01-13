# %%
import pandas as pd 
import numpy as np 

X = pd.DataFrame(np.random.normal(size=(100,100)),
                columns=['X_{}'.format(i+1) for i in range(100)])
X['Y'] = X['X_1'] + np.random.normal(size=100)

# %%
from scipy.stats import pearsonr

alpha = .05
n= len(X.columns) - 1
bonferroni_alpha = alpha/n
for xi in X.columns:
    r, p_value = pearsonr(X[xi], X['Y'])
    if p_value < bonferroni_alpha:
        print(xi, r, p_value, '***')
    elif p_value < alpha:
        print(xi, r, p_value)

# %%
