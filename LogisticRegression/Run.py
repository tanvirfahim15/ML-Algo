import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def sigmoid(vector):
    ret = []
    vector = vector * -1
    for i in vector:
        ret.append(math.exp(i))
    ret = np.ones(ret.__len__()) + ret
    ret = 1 / ret
    return ret




df=pd.read_csv('data.csv', sep=',',header=None)
df=df.values.flatten()
x1=[]
x2=[]
y=[]
for i in range(df.size):
    if i%3==0:
        x1.append(df[i])
    elif i%3==1:
        x2.append(df[i])
    else:
        y.append(df[i])
posx1=[]
posx2=[]
negx1=[]
negx2=[]
for i in range(x1.__len__()):
    if y[i]==1:
        posx1.append(x1[i])
        posx2.append(x2[i])
    else:
        negx1.append(x1[i])
        negx2.append(x2[i])

plt.scatter(posx1, posx2, color='y')
plt.scatter(negx1,negx2)
plt.show()

print(1)