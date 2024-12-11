from math import exp, sqrt, pi
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

def f(x, m, s):
    return 1 / sqrt(2 * pi * s ** 2) * exp(-(x - m) ** 2 / (2 * s ** 2))


m = 55.9
s = 25.9
xs = []
ys = []
sum_ = 0
Xs = []
ps = []


n = 1000
for X in range(1000):
    sum_ = 0
    X /= 1000
    Xs.append(X)

    for i in range(0, 200):
        xs.append(i)
        fx = f(i, m, s)
        ys.append(fx)
        sum_ += fx
        if sum_ >= X:
            if X == 0.8:
                print(i * 4.67)
            ps.append(i)
            break
    else:
        ps.append(i)


ps = [x * 4.67 for x in ps]

plt.plot(Xs, ps, color="mediumseagreen")
plt.xlabel(r"$X$")
plt.ylabel(r"$S(\lambda,P,X)$")
plt.title(r"$\lambda=4.67$")
plt.show()
