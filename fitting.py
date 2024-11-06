import matplotlib.pyplot as plt


data = {
    "Schiphol": (27_870_000, 61_000_000),
    "Lelystad": (12_000, 5_000_000),
    "Denver": (135_700_000, 77_000_000),
    "Zadar": (1000000,	1_500_000),
    "Manchester-Boston": (6_100_000, 2_000_000),
    "Calgary": (21_000_000, 18_500_000),
    "Lille": (4_500_000, 2_000_000),
    "Cologne": (10_000_000, 9_800_000),
    "Brussels": (12_450_000, 22_000_000),
}

y = [v[0] for v in data.values()]
x = [v[1] for v in data.values()]

plt.scatter(x, y)
for i, k in enumerate(data):
    plt.annotate(k, (x[i], y[i]))

plt.show()
