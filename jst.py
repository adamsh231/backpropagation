# %%
import numpy as np

# %%
def activation(x, treshold=0):
    return 1 if x >= treshold else 0

def andLogic(x):
    y_in = np.dot(x, [1, 1])
    return activation(y_in, 2)

def orLogic(x):
    y_in = np.dot(x, [2, 2])
    return activation(y_in, 2)

# %%
print(andLogic([1, 0]))
print(orLogic([1, 0]))

# %%
