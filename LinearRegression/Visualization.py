import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression


df = pd.read_csv('data.csv', sep=',', header=None)
arr = np.array(df)
arr = arr.flatten()
x = []
y = []
for i in range(0, arr.size, 2):
    x.append([arr[i]])
    y.append(arr[i + 1])

iterations = 1500
alpha = 0.01
theta = np.array([15.0, 16.0])

x = np.asarray(x)
y = np.asarray(y)

plt.scatter(x, y, color='k', marker='x')
plt.show()

ln = LinearRegression(theta, x, y, alpha)
cost_x = []
cost_y = []

for i in range(1500):
    ln.gradient_decent()
    cost_x.append(i)
    cost_y.append(ln.cost())
    if i % 100 == 0:
        plt.scatter(x, y, color='k', marker='x')
        x_axis = [0, 25]
        plt.plot(x_axis, ln.predict(np.array([[0], [25]])))
        plt.show()

print(ln.theta)

plt.plot(cost_x, cost_y)
plt.show()