#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
avgList = []
dData = []
nData = 100000
for i in range(1, nData, 100):
  dData.append(i)
  print(i)
  avg = 0
  iterator = i
  minI, maxI = 0 , i
  for j in range(iterator):
    number = np.random.randint(low=minI,high=maxI)
    min = minI
    max = maxI
    answer = np.random.randint(low=minI,high=maxI)
    count = 0
    while answer != number:
      count = count + 1
      if answer > number:
        max = answer
        answer = int((max - min) / 2) + min
      else:
        min = answer
        answer = int((max - min) / 2) + min
      if answer == number :
        count = count + 1
    avg = avg + count
  avg = avg/iterator
  avgList.append(avg)
  
#%%
plt.plot(avgList)
plt.ylabel('Average')
# plt.xticks(np.arange(len(dData)), dData)
plt.show()

#%%