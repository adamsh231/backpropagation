#%%
from backpropagationfix import *
import csv
from prettytable import PrettyTable

def minMaxNormalization(value, maks, mins):
    return ((value - mins)/(maks-mins))

#%%

#create pretty table
table = PrettyTable()
table.field_names = ["Epoch","MSE"]

# #read csv data
fileOpen = open('diabetesbp2.csv')
readFile = csv.reader(fileOpen)
data = list(readFile)
actual = list()
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])
    actual.append(data[i][-1])
    data[i].pop()

# #min-max normalization
dataTpose = np.transpose(data)
mins = [0 for i in range(len(data[0]))]
maks = [0 for i in range(len(data[0]))]
for i in range(len(maks)):
    maks[i] = max(dataTpose[i])
    mins[i] = min(dataTpose[i])
    
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = minMaxNormalization(data[i][j], maks[j], mins[j])

#create random weights, bias, alpha and epoch
bobot = 0.5
epoch = 10
alpha = 0.5
initial = initializeNetwork(8,2,9,1,bobot)
weights = initial[0]
bias = initial[1]
jmlData = 768

#plot data x
lblX = list()

#fixed weight and bias
fixWeights = fixBias = 0

#backpropagation function
for i in range(epoch):
    output = 0
    kebenaran = 0
    for j in range(jmlData):
        input = data[j]
        fact = [actual[j]]
        x = backpropagation(alpha,input,weights,fact,bias)
        weights = x[0]
        bias = x[1]
        output = x[2]
        fixWeights = weights
        fixBias = bias
        if x[3] == True:
            kebenaran += 1
    table.add_row([i, output])
    lblX.append(output)
    print("Epoch ("+str(i)+") Benar : "+str(kebenaran)+" Akurasi :"+str(float(kebenaran/jmlData*100))+"%")
print(table)

#%%
import matplotlib.pyplot as plt
plt.plot(lblX)
plt.ylabel('Mean Square Error')
plt.show()

#%%
for i in range(500,len(data)):
    input = data[i]
    fact = actual[i]
    hasil = feedForward(input,fixWeights,fixBias)
    nice = "";
    if closestValue(hasil[-1][0], [0,1]) == fact:
        nice = " Benar";
    print("Predict : "+str(hasil[-1])+" Target : "+str(int(fact))+nice)
    
#%%

# count = 0
# for i in range(jmlData, len(data)):
#     input = data[i]
#     fact = actual[i]
#     hasil = feedForward(input,fixWeights,fixBias)
#     if closestValue(hasil[len(fixWeights)][0],[1,0]) == int(fact):
#         count += 1
# print("Akurasi Data Uji: "+str(count/(768)*100)+" %")


#%%
j = initializeNetwork(8,1,12,2);

j[0]

# %%
