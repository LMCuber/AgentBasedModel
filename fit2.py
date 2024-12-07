
with open("nq4_data.txt") as f:
    a = [int(line.replace("\n", "").replace("\t", " ").split(" ")[1]) for line in f.readlines()]
    print(sum(a) / len(a))
    

import matplotlib.pyplot as plt



plt.style.use('ggplot')
plt.xticks([1, 2, 3, 4])

xs = [1, 2, 3, 4]
ys = [1.25666, 0.673333, 0.84, 1.13]
plt.plot(xs, ys, color="green")

plt.show()