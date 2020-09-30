import numpy as np
import random
from scipy.special import expit

#inisialisasi network bobot acak
def initializeNetwork(jmlInput, jmlHLayer, jmlHNeuron, jmlOutput, bobot = 0):
    weights = list()
    bias = list()
    if bobot == 0 :
        weights.append([[round(random.random(),2) for i in range(jmlInput)] for j in range(jmlHNeuron)])
        bias.append([0.5 for i in range(jmlHNeuron)])
        for i in range((jmlHLayer-1)):
            weights.append([[round(random.random(),2) for i in range(jmlHNeuron)] for j in range(jmlHNeuron)])
            bias.append([0.5 for i in range(jmlHNeuron)])
        weights.append([[round(random.random(),2) for i in range(jmlHNeuron)]for j in range(jmlOutput)])
        bias.append([0.5 for i in range(jmlOutput)])
    else:
        weights.append([[bobot for i in range(jmlInput)] for j in range(jmlHNeuron)])
        bias.append([0.5 for i in range(jmlHNeuron)])
        for i in range((jmlHLayer-1)):
            weights.append([[bobot for i in range(jmlHNeuron)] for j in range(jmlHNeuron)])
            bias.append([0.5 for i in range(jmlHNeuron)])
        weights.append([[bobot for i in range(jmlHNeuron)]for j in range(jmlOutput)])
        bias.append([0.5 for i in range(jmlOutput)])
    return [weights,bias]

#mengidentifikasi Network dari Weights
def detWeights(weights):
    jmlHLayer = len(weights) - 1
    jmlHNeuron = len(weights[1][0])
    jmlInput = len(weights[0][0])
    jmlOutput = len(weights[len(weights)-1])
    return [jmlInput,jmlHLayer,jmlHNeuron,jmlOutput]

#fungsi aktivasi sigmoid dengan 2 masukan array
def aktivasiSigBArray(input):
    return expit(input)
    # return [1 / (1 + np.exp(-x)) for x in input]

#fungsi aktivasi sigmoid dengan 2 masukan satuan
def aktivasiSigB(input):
    return expit(input)

#jarak terdekat
def closestValue(value, array):
    closest = array[0]
    jarak = abs(value - array[0])
    for i in range(len(array)):
        if jarak > abs(value - array[i]):
            jarak = abs(value - array[i])
            closest = array[i]
    return closest

# menghitung sum of square error MSE
def errorSquare(actual,predict):
    error = .0
    for i in range(len(predict)):
        error += (1/2)*pow((actual[i]-predict[i]),2)
    return error
    
#fase feedforward
def feedForward(input, weights, bias):
    outcome = 0
    hasil = list()
    hasil.append(input)
    for i in range(len(weights)):
        outcome = [0 for i in range(len(weights[i]))]
        for j in range(len(weights[i])):
            outcome[j] = np.dot(input, weights[i][j]) + bias[i][j]
            outcome[j] = aktivasiSigB(outcome[j])
        input = outcome
        hasil.append(input)
    return hasil

#backpropagation
def backpropagation(alpha ,input ,weights, actual, bias):
    network = detWeights(weights)
    initial = initializeNetwork(network[0],network[1],network[2],network[3])
    delWeights = initial[0]
    hasil = feedForward(input,weights, bias)
    mse = errorSquare(actual,hasil[len(weights)])
    
    #PERCOBAAN UNTUK OUTPUT NYA HANYA 1, JIKA OUTPUT LEBIH DARI SATU GANTI MATRIK
    prediksi = hasil[len(weights)][0] #hasil isinya matrik [x,y,...,n]
    # nice = "";
    # if closestValue(prediksi, [0,1]) == actual[0]:
    #     nice = " Benar";
    # print("Prediksi : ("+str(closestValue(prediksi, [0,1]))+") Target : "+str(int(actual[0]))+nice)
    nilai0 = prediksi
    nilai1 = 1 - prediksi
    kebenaran = True
    if nilai0 <= nilai1 :
        prediksi = 0
    else:
        prediksi = 1
    if prediksi != actual[0]:#actual isinya matrik [x,y,...,n]
        kebenaran = False
    
    for i in range(len(weights[-1])):
        hasil[-1][i] = (actual[i] - hasil[-1][i])*(hasil[-1][i] * (1 - hasil[-1][i]))
    for i in reversed(range(len(weights))):
        tpose = np.transpose(weights[i])
        for j in range(len(weights[i])):
            delBias = 0
            for k in range(len(weights[i][j])):
                delWeights[i][j][k] = alpha * hasil[(i+1)][j] * hasil[i][k]
                delBias = alpha * hasil[(i+1)][j]
                hasil[i][k] = np.dot(tpose[k], hasil[(i+1)])
            bias[i][j] += delBias
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                weights[i][j][k] += delWeights[i][j][k]
    return [weights,bias,mse,kebenaran]