from matplotlib import pyplot as plt
import numpy as np
from math import sqrt, exp, pi
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from numpy import array
from random import random


data = [12.643333, 9.916667, 10.886667, 11.118333, 12.258333, 5.446667, 11.365000, 14.355000, 18.128333, 11.600000, 9.886667, 4.730000, 12.710000, 22.676667, 7.486667, 5.010000, 13.778333, 13.640000, 13.946667, 12.041667, 11.416667, 12.666667, 14.995000, 10.393333, 6.428333, 4.488333, 6.111667, 5.418333, 5.643333, 15.103333, 15.868333, 5.966667, 11.763333, 12.621667, 5.426667, 23.811667, 31.736667, 10.471667, 7.498333, 11.495000, 4.475000, 11.610000, 10.071667, 21.441667, 6.981667, 10.375000, 34.795000, 10.685000, 14.013333, 5.316667, 13.705000, 11.763333, 6.245000, 12.685000, 6.331667, 33.720000, 20.000000, 17.536667, 34.891667, 13.676667, 16.488333, 12.585000, 7.835000, 20.623333, 5.685000, 6.691667, 4.613333, 19.238333, 5.060000, 24.775000, 6.121667, 14.083333, 22.593333, 6.281667, 14.293333, 6.996667, 8.250000, 36.188333, 37.910000, 13.953333, 10.808333, 15.141667, 12.560000, 8.040000, 12.793333, 13.906667, 12.925000, 13.528333, 6.371667, 7.288333, 18.665000, 34.361667, 34.285000, 8.466667, 6.148333, 15.765000, 13.525000, 6.151667, 38.651667, 22.583333, 4.358333, 11.708333, 15.075000, 12.521667, 6.326667, 13.978333, 11.140000, 12.386667, 5.396667, 15.610000, 16.751667, 5.473333, 6.245000, 14.348333, 12.858333, 10.935000, 13.925000, 5.451667, 4.808333, 38.713333, 24.106667, 10.595000, 36.810000, 13.571667, 14.196667, 5.571667, 11.795000, 19.175000, 13.671667, 34.385000, 12.363333, 4.500000, 7.943333, 15.360000, 5.488333, 36.508333, 12.810000, 7.098333, 21.131667, 14.723333, 17.231667, 13.256667, 11.253333, 17.100000, 9.821667, 11.815000, 41.781667, 8.481667, 9.816667, 13.735000, 17.120000, 14.963333, 5.045000, 34.538333, 17.845000, 13.126667, 11.600000, 13.576667, 11.531667, 34.631667, 39.030000, 4.793333, 35.780000, 20.623333, 20.243333, 11.598333, 44.528333, 34.951667, 33.861667, 13.280000, 32.598333, 13.788333, 4.881667, 37.993333, 13.678333, 15.173333, 13.663333, 36.275000, 12.406667, 16.693333, 35.625000, 16.046667, 33.951667, 20.315000, 42.266667, 43.120000, 40.601667, 34.160000, 14.640000, 44.756667, 32.778333, 13.008333, 11.738333, 13.226667, 24.435000, 19.996667, 37.681667, 5.800000, 23.670000, 8.151667, 5.615000, 7.158333, 16.161667, 12.735000, 32.345000, 6.050000, 11.531667, 17.958333, 15.470000, 35.396667, 15.513333, 10.181667, 32.826667, 42.735000, 38.628333, 34.325000, 9.283333, 16.205000, 16.500000, 33.686667, 23.296667, 15.665000, 42.660000, 65.235000, 21.600000, 44.795000, 6.125000, 18.678333, 44.015000, 16.390000, 6.935000, 42.318333, 14.340000, 8.731667, 41.530000, 5.830000, 26.010000, 33.976667, 14.800000, 6.586667, 7.788333, 14.681667, 36.618333, 62.896667, 9.500000, 37.171667, 41.286667, 18.590000, 22.051667, 20.801667, 35.310000, 14.346667, 20.735000, 13.131667, 18.311667, 18.196667, 22.396667, 35.700000, 8.011667, 15.771667, 7.461667, 63.776667, 9.950000, 12.655000, 7.156667, 8.898333, 7.666667, 44.893333, 12.571667, 50.991667, 15.646667, 13.626667, 37.808333, 15.228333, 21.338333, 39.193333, 22.550000, 46.686667, 17.165000, 37.971667, 19.646667, 25.078333, 37.173333, 40.555000, 9.110000, 7.915000, 8.691667, 14.173333, 20.380000, 38.970000, 41.231667, 33.036667, 19.630000, 41.201667, 45.446667, 27.000000, 20.180000, 47.016667, 13.130000, 15.856667, 26.660000, 41.430000, 13.610000, 33.181667, 34.255000, 16.073333, 15.248333, 7.260000, 25.860000, 18.411667, 7.726667, 36.000000, 20.371667, 18.831667, 22.661667, 36.893333, 33.071667, 13.815000, 42.888333, 45.195000, 9.243333, 8.428333, 14.406667, 12.668333, 16.253333, 36.475000, 44.806667, 12.500000, 17.235000, 23.305000, 16.065000, 6.613333, 15.156667, 16.288333, 8.025000, 14.190000, 8.158333, 34.145000, 20.628333, 45.948333, 16.986667, 20.365000, 39.901667, 41.361667, 45.008333, 14.578333, 14.531667, 44.545000, 24.356667, 32.245000, 50.843333, 9.146667, 42.858333, 8.050000, 16.671667, 23.105000, 40.498333, 7.756667, 71.630000, 17.738333, 46.678333, 32.660000, 20.643333, 23.595000, 44.605000, 15.146667, 44.756667, 39.026667, 38.861667, 15.050000, 37.255000, 15.925000, 15.516667, 24.038333, 25.791667, 45.785000, 8.308333, 35.888333, 38.153333, 43.046667, 8.221667, 17.265000, 45.073333, 15.583333, 38.608333, 23.683333, 14.151667, 36.148333, 38.835000, 23.636667, 64.345000, 14.020000, 36.158333, 44.701667, 37.851667, 23.958333, 14.495000, 15.528333, 48.633333, 7.203333, 10.896667, 16.741667, 10.050000, 30.523333, 20.331667, 16.396667, 13.535000, 34.896667, 23.435000, 47.095000, 25.773333, 18.045000, 16.911667, 16.600000, 28.701667, 19.686667, 11.378333, 71.830000, 26.106667, 48.815000, 12.780000, 23.040000, 13.145000, 34.295000, 10.270000, 26.665000, 26.563333, 46.023333, 70.820000, 10.721667, 49.490000, 11.696667, 23.895000, 46.626667, 39.583333, 24.593333, 61.220000, 25.911667, 7.571667, 45.163333, 13.900000, 22.780000, 9.170000, 9.038333, 9.350000, 25.575000, 22.075000, 17.878333, 38.853333, 19.151667, 44.465000, 9.083333, 8.816667, 22.871667, 20.525000, 43.956667, 40.205000, 39.425000, 21.025000, 15.496667, 51.240000, 15.101667, 24.006667, 44.520000, 17.033333, 25.648333, 42.525000, 10.335000, 43.290000, 47.826667, 39.246667, 37.608333, 10.958333, 16.038333, 21.405000, 14.205000, 24.558333, 24.123333, 48.583333, 8.670000, 17.180000, 13.923333, 42.311667, 23.161667, 16.796667, 10.843333, 46.663333, 9.601667, 21.530000, 36.266667, 20.723333, 11.058333, 32.033333, 39.766667, 20.271667, 11.536667, 18.640000, 27.420000, 11.828333, 40.838333, 56.395000, 9.608333, 36.568333, 20.508333, 65.403333, 45.213333, 37.275000, 14.310000, 11.375000, 19.123333, 19.716667, 18.040000, 18.800000, 46.960000, 21.196667, 78.291667, 44.115000, 22.445000, 18.631667, 18.675000, 23.258333, 12.988333, 20.211667, 46.093333, 12.078333, 18.933333, 16.743333, 39.335000, 9.990000, 47.790000, 20.438333, 50.116667, 41.756667, 53.531667, 56.685000, 9.776667, 12.721667, 24.715000, 17.646667, 19.603333, 51.073333, 37.498333, 14.785000, 44.873333, 13.161667, 26.180000, 47.526667, 12.885000, 24.123333, 42.445000, 32.556667, 16.570000, 51.671667, 38.775000, 38.688333, 23.456667, 13.160000, 19.211667, 18.320000, 12.130000, 10.316667, 11.990000, 16.655000, 42.340000, 14.215000, 18.651667, 36.778333, 41.270000, 12.891667, 40.295000, 52.540000, 25.080000, 46.960000, 11.060000, 41.131667, 19.548333, 43.838333, 68.858333, 19.893333, 42.725000, 53.073333, 10.530000, 10.840000, 66.361667, 64.876667, 30.276667, 36.530000, 19.388333, 47.355000, 28.468333, 38.261667, 44.411667, 24.726667, 18.498333, 54.981667, 42.801667, 38.303333, 25.455000, 54.073333, 80.248333, 13.223333, 40.861667, 11.781667, 50.400000, 17.961667, 39.736667, 11.878333, 26.183333, 29.166667, 32.526667, 51.711667, 19.568333, 31.908333, 47.230000, 43.170000, 32.455000, 42.381667, 11.275000, 33.451667, 47.193333, 18.680000, 43.583333, 37.121667, 26.421667, 58.096667, 11.011667, 46.920000, 40.651667, 17.255000, 34.891667, 55.850000, 16.936667, 52.178333, 12.350000, 12.268333, 16.346667, 11.960000, 11.255000, 56.056667, 23.255000, 20.468333, 43.605000, 48.801667, 21.618333, 68.148333, 51.286667, 22.501667, 19.118333, 38.130000, 11.821667, 19.661667, 11.593333, 12.670000, 72.808333, 59.696667, 68.890000, 16.088333, 52.666667, 20.533333, 13.580000, 12.088333, 11.431667, 27.105000, 42.056667, 32.216667, 19.770000, 53.011667, 40.610000, 44.468333, 42.891667, 12.688333, 48.615000, 21.376667, 38.800000, 30.048333, 41.778333, 31.320000, 52.995000, 12.495000, 75.818333, 40.508333, 12.918333, 27.268333, 13.141667, 42.678333, 18.451667, 11.745000, 29.820000, 25.465000, 83.458333, 11.961667, 13.618333, 37.866667, 34.221667, 68.588333, 21.813333, 18.950000, 16.841667, 24.971667, 49.283333, 51.900000, 23.535000, 39.975000, 16.153333, 23.530000, 49.726667, 56.536667, 50.250000, 38.670000, 25.443333, 34.625000, 40.291667, 53.743333, 52.726667, 13.743333, 40.216667, 13.898333, 43.938333, 53.665000, 50.803333, 30.071667, 11.980000, 25.445000, 47.465000, 14.825000, 22.013333, 34.266667, 18.323333, 11.971667, 12.616667, 16.946667, 13.033333, 17.990000, 29.581667, 25.608333, 34.381667, 41.953333, 44.625000, 12.630000, 36.841667, 20.511667, 32.691667, 42.660000, 52.225000, 57.013333, 17.611667, 11.945000, 64.708333, 42.140000, 19.233333, 45.505000, 47.600000, 25.976667, 25.770000, 19.561667, 19.390000, 45.346667, 12.641667, 46.326667, 39.596667, 37.670000, 28.415000, 12.480000, 44.233333, 72.696667, 21.030000, 53.293333, 13.495000, 38.188333, 13.805000, 13.053333, 27.260000, 11.336667, 41.630000, 43.606667, 41.440000, 37.000000, 47.593333, 33.185000, 11.016667, 29.851667, 71.965000, 65.945000, 34.861667, 20.215000, 18.581667, 53.208333, 39.990000, 13.090000, 13.440000, 47.471667, 22.345000, 33.435000, 70.688333, 87.905000, 48.968333, 12.273333, 32.700000, 41.813333, 30.603333, 44.785000, 21.730000, 27.221667, 20.380000, 14.570000, 20.603333, 13.206667, 50.066667, 20.485000, 58.243333, 14.358333, 34.900000, 41.308333, 41.816667, 34.085000, 12.911667, 14.188333, 58.415000, 13.885000, 14.690000, 13.741667, 52.726667, 39.261667, 14.790000, 19.946667, 33.860000, 62.711667, 13.471667, 45.255000, 50.150000, 28.356667, 45.073333, 56.486667, 22.883333, 21.736667, 43.026667, 18.946667, 42.608333, 55.560000, 20.881667, 18.170000, 44.731667, 52.806667, 52.136667, 30.563333, 41.578333, 68.341667, 62.660000, 18.418333, 30.160000, 13.656667, 40.796667, 41.048333, 71.096667, 71.551667, 21.565000, 45.808333, 21.995000, 21.513333, 62.388333, 16.401667, 15.383333, 25.316667, 29.600000, 36.340000, 73.725000, 27.070000, 13.121667, 20.886667, 29.443333, 53.500000, 48.903333, 41.183333, 19.088333, 48.386667, 22.230000, 20.901667, 25.335000, 39.606667, 14.761667, 41.768333, 56.111667, 63.595000, 12.136667, 21.738333, 18.610000, 12.431667, 12.605000, 41.153333, 56.336667, 20.585000, 12.476667, 35.703333, 55.123333, 49.473333, 22.616667, 29.795000, 58.238333, 10.620000, 29.158333, 51.058333, 36.325000, 19.130000, 60.468333, 14.216667, 18.403333, 22.428333, 22.663333, 51.651667, 56.938333, 19.631667, 12.035000, 46.730000, 44.500000, 48.715000, 28.875000, 42.161667, 29.495000, 43.815000, 34.331667, 21.535000, 45.175000, 12.758333, 44.090000, 49.153333, 60.285000, 39.841667, 17.390000, 53.791667, 33.348333, 20.153333, 40.091667, 45.053333, 12.535000, 53.411667, 72.465000, 50.520000, 13.011667, 51.593333, 21.263333, 61.551667, 27.920000, 19.688333, 55.600000, 39.793333, 71.703333, 50.846667, 21.175000, 32.588333, 35.881667, 41.775000, 18.691667, 19.990000, 65.966667, 34.215000, 42.905000, 44.146667, 11.686667, 54.761667, 12.546667, 20.405000, 48.783333, 21.690000, 18.366667, 17.570000, 12.440000, 70.250000, 43.181667, 47.508333, 46.298333, 85.975000, 41.753333, 33.645000, 23.318333, 19.570000, 10.961667, 52.671667, 43.663333, 67.993333, 19.095000, 15.738333, 19.305000, 44.321667, 23.758333, 34.273333, 34.665000, 30.198333, 38.953333, 89.633333, 63.440000, 62.346667, 15.448333, 31.156667, 21.105000, 44.471667, 69.601667, 27.455000, 56.778333, 17.705000, 28.451667, 53.470000, 41.800000, 53.401667, 37.543333, 12.510000, 42.985000, 36.255000, 26.403333, 26.636667, 34.740000, 90.496667, 13.816667, 75.470000, 42.281667, 20.858333, 19.798333, 61.506667, 50.001667, 43.590000, 11.893333, 42.286667, 15.163333, 35.913333, 60.606667, 16.825000, 60.975000, 44.378333, 51.240000, 50.200000, 17.963333, 20.140000, 60.453333, 23.236667, 48.683333, 59.565000, 10.660000, 43.528333, 19.130000, 74.478333, 92.990000, 36.098333, 39.730000, 42.601667, 15.945000, 36.921667, 49.666667, 42.165000, 53.086667, 28.415000, 44.645000, 68.785000, 35.496667, 68.355000, 11.833333, 36.905000, 65.513333, 69.221667, 53.423333, 41.075000, 49.868333, 13.988333, 52.396667, 17.423333, 23.396667, 12.240000, 26.636667, 37.308333, 48.298333, 54.953333, 46.906667, 59.745000, 12.705000, 48.865000, 52.556667, 50.565000, 76.641667, 56.790000, 15.465000, 43.976667, 47.613333, 42.126667, 44.210000, 69.696667, 31.406667, 51.906667, 16.510000, 51.188333, 25.921667, 70.936667, 39.423333, 35.956667, 17.046667, 40.591667, 46.381667, 48.880000, 47.610000, 55.236667, 37.070000, 42.910000, 67.078333, 41.746667, 42.703333, 20.855000, 34.090000, 39.956667, 72.400000, 47.686667, 51.530000, 33.345000, 40.453333, 71.041667, 42.235000, 35.553333, 69.881667, 72.991667, 14.485000, 18.480000, 35.898333, 42.728333, 16.768333, 37.978333, 33.891667, 48.605000, 61.923333, 33.786667, 40.135000, 12.998333, 45.151667, 41.025000, 44.015000, 47.620000, 65.086667, 35.421667, 42.440000, 36.731667, 39.938333, 72.110000, 35.151667, 47.663333, 68.878333, 39.566667, 40.660000, 42.493333, 47.203333, 44.168333, 71.918333, 60.348333, 62.693333, 49.260000, 39.506667, 45.800000, 54.845000, 63.266667, 54.140000, 64.168333, 85.303333, 45.733333, 66.066667, 67.626667, 65.890000, 64.148333]

plt.style.use('ggplot')


m_a, s_a = 15, 5
A = lambda x: 1 / np.sqrt(2 * pi * s_a ** 2) * np.exp(-(x - m_a) ** 2 / (2 * s_a ** 2))
m_b, s_b = 44, 5
B = lambda x: 1 / np.sqrt(2 * pi * s_b ** 2) * np.exp(-(x - m_b) ** 2 / (2 * s_b ** 2))
m_c, s_c = 68, 2.5
C = lambda x: 1 / np.sqrt(2 * pi * s_c ** 2) * np.exp(-(x - m_c) ** 2 / (2 * s_c ** 2))
k_a, k_b, k_c = 730, 330, 50
T = lambda x: k_a * A(x) + k_b * B(x) + k_c * C(x)
P = lambda x: T(x) / quad(T, 0, 100)[0]

rang = 1000

xs = np.linspace(0, 100, rang)

cdf_values = np.cumsum(P(xs) * np.diff(xs, prepend=0))
cdf_interpolator = interp1d(cdf_values, xs, kind="linear", bounds_error=False, fill_value="extrapolate")

us = [cdf_interpolator(u / rang) for u in range(rang)]

print(sum(us) / len(us))

def model_function(x, a, b, c, d, e, f, g, h, i):
    return a / (1 + np.exp(-b * (x - c))) + d / (1 + np.exp(-e * (x - f))) + g / (1 + np.exp(-h * (x - i)))

params, covariance = curve_fit(model_function, xs, us, maxfev=1000000000)

y_fit = model_function(xs, *params)

plt.plot(xs, us, label="Actual")
plt.plot(xs, y_fit, label="Fit")
plt.legend()
plt.show()