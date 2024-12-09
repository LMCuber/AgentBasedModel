from math import exp, sqrt, pi
import matplotlib.pyplot as plt


def f(x, m, s):
    return 1 / sqrt(2 * pi * s ** 2) * exp(-(x - m) ** 2 / (2 * s ** 2))


r = 120
xs = list(range(r))
m = 62
s = 21
ys = [f(i, m, s) for i in range(r)]

sum_ = 0
X = 0.5
d = 10000
for i in range(int(r * d)):
    i /= d
    sum_ += 1 / d * f(i, m, s)
print(sum_)

plt.plot(xs, ys)
plt.show()
