# %%
import numpy as np

def sig(X):
    return [1 / (1 + np.exp(-x)) for x in X]

def sigd(X):
    for i, x in enumerate(X):
        s = sig([x])[0]

        yield s * (1 - s)

def bin_enc(lbl):
    mi = min(lbl)
    length = len(bin(max(lbl) - mi + 1)[2:])
    enc = []

    for i in lbl:
        b = bin(i - mi)[2:].zfill(length)

        enc.append([int(n) for n in b])

    return enc

def bin_dec(enc, mi=0):
    lbl = []

    for e in enc:
        rounded = [int(round(x)) for x in e]
        string = ''.join(str(x) for x in rounded)
        num = int(string, 2) + mi

        lbl.append(num)

    return lbl
  
def onehot_enc(lbl, min_val=0):
    mi = min(lbl)
    enc = np.full((len(lbl), max(lbl) - mi + 1), min_val, np.int8)

    for i, x in enumerate(lbl):
        enc[i, x - mi] = 1

    return enc

def onehot_dec(enc, mi=0):
    return [np.argmax(e) + mi for e in enc]

def bp_fit(C, X, t, a, mep, mer):
    nin = [np.empty(i) for i in C]
    n = [np.empty(j + 1) if i < len(C) - 1 else np.empty(j) for i, j in enumerate(C)]
    w = np.array([np.random.rand(C[i] + 1, C[i + 1]) for i in range(len(C) - 1)])
    dw = [np.empty((C[i] + 1, C[i + 1])) for i in range(len(C) - 1)]
    d = [np.empty(s) for s in C[1:]]
    din = [np.empty(s) for s in C[1:-1]]
    ep = 0
    mse = 1

    for i in range(0, len(n) - 1):
        n[i][-1] = 1

    while (mep == -1 or ep < mep) and mse > mer:
        ep += 1
        mse = 0

        for r in range(len(X)):
            n[0][:-1] = X[r]

            for L in range(1, len(C)):
                nin[L] = np.dot(n[L - 1], w[L - 1])

                n[L][:len(nin[L])] = sig(nin[L])

            e = t[r] - n[-1]
            mse += sum(e ** 2)
            d[-1] = e * list(sigd(nin[-1]))
            dw[-1] = a * d[-1] * n[-2].reshape((-1, 1))

            for L in range(len(C) - 1, 1, -1):
                din[L - 2] = np.dot(d[L - 1], np.transpose(w[L - 1][:-1]))
                d[L - 2] = din[L - 2] * np.array(list(sigd(nin[L - 1])))
                dw[L - 2] = (a * d[L - 2]) * n[L - 2].reshape((-1, 1))

            w += dw

        mse /= len(X)
        
        if ep % 50 == 0:
            print(f'Epoch #{ep}, MSE: {mse}')

    return w, ep, mse

def bp_predict(X, w):
    n = [np.empty(len(i)) for i in w]
    nin = [np.empty(len(i[0])) for i in w]
    predict = []

    n.append(np.empty(len(w[-1][0])))

    for x in X:
        n[0][:-1] = x

        for L in range(0, len(w)):
            nin[L] = np.dot(n[L], w[L])
            n[L + 1][:len(nin[L])] = sig(nin[L])

        predict.append(n[-1].copy())

    return predict

#%%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = minmax_scale(iris.data)
Y = onehot_enc(iris.target)

c = 4, 9, 3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)
w, ep, mse = bp_fit(c, X_train, y_train, .1, 1000, .1)

print(f'Epoch: {ep}')
print(f'MSE: {mse}')

predict = bp_predict(X_test, w)
predict = onehot_dec(predict)
y_test = onehot_dec(y_test)
acc = accuracy_score(predict, y_test)

print(f'Output: {predict}')
print(f'True  : {y_test}')
print(f'Accuracy: {acc}')

#%%
