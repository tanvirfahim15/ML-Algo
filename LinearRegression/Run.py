import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

def hypo(theta, x):
    return theta[0]+theta[1]*x


def cost(theta, x, y):
    m = x.__len__()
    ret =float(0)
    for i in range(m):
        h=(hypo(theta, x[i])-y[i])
        ret = ret+h*h
    ret = ret/(2*m)
    return ret


def batch(theta, x, y, alpha):
    m = x.__len__()
    off=float(0)
    ret=[0.0,0.0]
    for i in range(m):
        off=off+hypo(theta,x[i])-y[i]
    ret[0]=theta[0]-off*(alpha/m)
    off=0.0
    for i in range(m):
        off=off+(hypo(theta,x[i])-y[i])*x[i]
    ret[1]=theta[1]-off*(alpha/m)
    return ret


df = pd.read_csv('data.csv',sep=',')
arr = np.array(df)
arr=arr.flatten()
x=[]
y=[]
for i in range(0,arr.size,2):
    x.append(arr[i])
    y.append(arr[i+1])



iterations = 1500
alpha = 0.01
theta=np.array([10.0 ,12.0])
plt.scatter(x, y, color='k', marker='x')
plt.show()

print(cost(theta,x,y))

it_x=[]
it_y=[]
for i in range(iterations):
    theta=batch(theta,x,y,alpha)
    it_x.append(i)
    it_y.append(cost(theta,x,y))


print(cost(theta,x,y))
print(theta)


plt.scatter(x, y, color='k', marker='x')
x1=[-1,20]
y1=[hypo(theta,1),hypo(theta,2)]

plt.plot(x1, y1)
plt.show()

plt.plot(it_x,it_y)
plt.show()