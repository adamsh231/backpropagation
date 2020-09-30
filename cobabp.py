# %%
import numpy as np

#%%
w = [3,3] #bobot awal

def aktivasi(x):
    return (1/(1+np.exp(np.dot(-1,x))))
    

def inputLayer(x):
    x_in = np.dot(x,w)
    return aktivasi([x_in])
    
#%%
print(inputLayer([0.3,4]))

#%%
