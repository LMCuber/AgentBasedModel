import matplotlib.pyplot as plt
from pprint import pprint


datas = []
with open("arrivals.txt") as f:
    prev_line = None
    for i, line in enumerate(f.readlines()):
        line = line.removesuffix("\n")
        if line:
            line = float(line)
            if prev_line is None or line < prev_line:
                datas.append([line])
            else:
                datas[-1].append(line)
            prev_line = line

cumulative = []
freqs = []
for i, data in enumerate(datas):
    for m in range(0, 300):
        cumulative.append(len([d for d in data if d <= m]))
        if m == 0:
            freqs.append(cumulative[-1])
        else:
            freqs.append(cumulative[-1] - cumulative[-2])

print(sum(freqs) / len(freqs))

plt.hist(freqs, bins=100)
plt.show()