# %%
import numpy as np
import csv
import random
from prettytable import PrettyTable

# %%
fileOpen = open('diabetes.csv')
readFile = csv.reader(fileOpen)
data = list(readFile)
outcome = [0 for i in range(len(data))]
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])
    outcome[i] = data[i][8]
    data[i].pop()

# %%
jmlFitur = 8
jmlNeuronHidden = 6
jmlBias = jmlNeuronHidden + 1
jmlBobot = jmlFitur * jmlNeuronHidden + jmlNeuronHidden + jmlBias
alpha = 0.7

# inisialisasi bobot ,value aktivasi dan bias
bobot1 = [[random.random() for i in range(jmlFitur)]
          for j in range(jmlNeuronHidden)]
bias1 = [random.random() for i in range(jmlNeuronHidden)]
bobot2 = [random.random() for i in range(jmlNeuronHidden)]
bias2 = random.random()

deltaBobot1 = [[0 for i in range(jmlFitur)] for j in range(jmlNeuronHidden)]
deltaBobot2 = [0 for i in range(jmlNeuronHidden)]
deltaBias1 = [0 for i in range(jmlNeuronHidden)]
deltaBias2 = 0

valueHidden = [0 for i in range(jmlNeuronHidden)]
valueOutput = 0

# %%


def aktivasi(x):
    valueOutput = (1/(1+np.exp(np.dot(-1, x))))
    return valueOutput


def inputLayer(count):
    for i in range(jmlNeuronHidden):
        x_in = np.dot(data[count], bobot1[i]) + bias1[i]
        valueHidden[i] = aktivasi(x_in)
    return hiddenLayer(valueHidden, count)


def hiddenLayer(x, count):
    y_in = np.dot(x, bobot2) + bias2
    return backProp(aktivasi(y_in), count, bias2)

# %%


def backProp(x, count, y):
    valueOutput = (outcome[count] - x) * (x * (1 - x))
    for i in range(jmlNeuronHidden):
        deltaBobot2[i] = alpha * valueOutput * valueHidden[i]
        deltaBias2 = alpha * valueOutput * 1
        valueHidden[i] = valueOutput * bobot2[i]
    for i in range(jmlNeuronHidden):
        for j in range(jmlFitur):
            deltaBobot1[i][j] = alpha * valueHidden[i] * data[i][j]
        deltaBias1[i] = alpha * valueHidden[i] * 1
    for i in range(jmlNeuronHidden):
        for j in range(jmlFitur):
            bobot1[i][j] = bobot1[i][j] + deltaBobot1[i][j]
        bobot2[i] = bobot2[i] + deltaBobot2[i]
        bias1[i] = bias1[i] + deltaBias1[i]
    bias2 = y + deltaBias2

# %%


for i in range(500):
    inputLayer(i)
    



# %%
table = [PrettyTable() for i in range(2)]
for i in range(jmlNeuronHidden):
    table[0].add_row(bobot1[i])
table[1].add_row(bobot2)

print(table[0])
print(table[1])

#%%
