import matplotlib.pyplot as plt


data = [130.180000, 115.625000, 113.135000, 112.050000, 111.080000, 110.235000, 109.880000, 109.445000, 108.650000, 108.395000, 106.345000, 104.420000, 103.935000, 103.760000, 103.025000, 101.945000, 101.355000, 100.140000, 129.160000, 98.200000, 96.040000, 95.700000, 124.550000, 93.745000, 92.830000, 92.645000, 92.415000, 92.250000, 121.590000, 91.370000, 120.755000, 90.435000, 120.395000, 89.515000, 119.430000, 87.060000, 87.025000, 86.945000, 86.525000, 85.205000, 85.150000, 84.850000, 114.570000, 84.535000, 83.445000, 83.210000, 83.100000, 82.560000, 82.205000, 112.105000, 81.755000, 81.465000, 81.195000, 80.610000, 78.835000, 78.700000, 108.625000, 78.410000, 78.245000, 78.200000, 107.760000, 77.755000, 77.500000, 107.135000, 76.940000, 76.925000, 76.320000, 76.250000, 76.015000, 75.190000, 75.120000, 105.075000, 104.965000, 74.850000, 73.950000, 73.935000, 73.815000, 103.740000, 73.610000, 73.305000, 103.300000, 103.215000, 73.115000, 73.035000, 72.610000, 72.375000, 72.050000, 71.295000, 71.065000, 70.870000, 70.570000, 69.565000, 69.255000, 69.220000, 99.185000, 68.910000, 68.805000, 68.595000, 68.540000, 68.520000, 68.435000, 67.890000, 97.730000, 127.320000, 67.170000, 66.700000, 66.575000, 66.450000, 66.365000, 125.930000, 95.850000, 65.705000, 65.660000, 65.535000, 95.395000, 65.300000, 65.175000, 65.050000, 94.740000, 94.650000, 64.550000, 64.490000, 94.235000, 63.535000, 93.125000, 93.055000, 63.035000, 92.755000, 62.175000, 62.135000, 91.835000, 61.700000, 61.360000, 61.260000, 60.465000, 60.255000, 60.165000, 90.165000, 119.635000, 89.610000, 59.155000, 59.080000, 58.880000, 88.815000, 58.770000, 58.720000, 88.325000, 58.025000, 87.910000, 57.900000, 87.570000, 57.180000, 56.440000, 56.160000, 116.110000, 56.010000, 55.685000, 55.630000, 55.490000, 55.290000, 55.005000, 54.670000, 54.650000, 114.610000, 54.315000, 54.025000, 83.860000, 53.630000, 113.270000, 52.985000, 52.915000, 52.755000, 82.495000, 82.465000, 52.405000, 52.305000, 52.225000, 81.880000, 81.825000, 111.725000, 51.705000, 51.670000, 51.575000, 51.115000, 80.670000, 80.425000, 50.375000, 50.100000, 80.075000, 79.745000, 49.665000, 49.430000, 79.160000, 49.135000, 48.690000, 78.640000, 78.330000, 48.295000, 78.200000, 108.145000, 48.080000, 77.915000, 47.820000, 77.760000, 77.290000, 77.105000, 76.990000, 46.790000, 46.770000, 76.735000, 76.065000, 75.620000, 45.500000, 45.495000, 45.430000, 45.145000, 74.740000, 74.695000, 44.505000, 74.275000, 103.960000, 43.745000, 73.680000, 73.620000, 73.520000, 103.420000, 73.310000, 42.920000, 72.670000, 72.580000, 72.430000, 72.350000, 102.245000, 71.990000, 71.615000, 71.530000, 70.895000, 100.825000, 70.380000, 40.320000, 40.240000, 70.125000, 69.950000, 69.375000, 39.250000, 38.990000, 68.900000, 68.575000, 68.410000, 38.355000, 68.130000, 68.010000, 127.885000, 37.645000, 37.450000, 37.135000, 66.920000, 96.180000, 66.120000, 36.105000, 35.755000, 65.555000, 95.550000, 65.500000, 65.405000, 65.315000, 65.200000, 34.915000, 34.875000, 64.855000, 94.470000, 34.335000, 64.280000, 94.080000, 33.685000, 63.585000, 33.545000, 33.335000, 93.225000, 63.215000, 32.990000, 62.890000, 62.680000, 92.485000, 32.440000, 32.405000, 92.190000, 61.960000, 61.935000, 61.290000, 61.280000, 31.240000, 91.205000, 91.075000, 30.815000, 60.660000, 60.420000, 30.025000, 59.910000, 119.540000, 29.510000, 29.380000, 29.270000, 29.215000, 58.815000, 88.485000, 88.175000, 87.780000, 87.560000, 87.515000, 57.465000, 117.155000, 87.025000, 26.760000, 86.715000, 56.535000, 26.310000, 56.235000, 55.675000, 85.525000, 55.445000, 25.290000, 55.130000, 114.885000, 54.780000, 84.670000, 24.430000, 54.380000, 54.335000, 54.325000, 54.155000, 54.130000, 23.980000, 23.540000, 53.475000, 83.205000, 53.140000, 52.945000, 52.500000, 52.490000, 81.420000, 51.210000, 21.190000, 51.070000, 51.045000, 50.700000, 80.695000, 80.515000, 80.265000, 80.100000, 79.950000, 49.835000, 49.715000, 49.600000, 49.440000, 79.410000, 48.895000, 78.770000, 48.645000, 78.525000, 48.395000, 48.255000, 78.050000, 47.525000, 17.330000, 17.225000, 16.975000, 106.855000, 16.805000, 46.480000, 76.460000, 76.305000, 46.150000, 16.070000, 76.070000, 75.900000, 45.810000, 45.800000, 45.720000, 75.690000, 75.495000, 14.590000, 74.380000, 44.370000, 74.350000, 74.285000, 74.165000, 44.025000, 103.935000, 73.920000, 73.465000, 43.455000, 42.630000, 42.525000, 42.265000, 41.935000, 71.870000, 71.790000, 101.775000, 41.665000, 41.535000, 71.490000, 41.140000, 70.835000, 40.750000, 70.370000, 70.295000, 9.955000, 69.865000, 69.795000, 39.750000, 69.345000, 39.290000, 39.170000, 9.145000, 68.870000, 68.475000, 68.440000, 68.360000, 68.215000, 38.190000, 67.460000, 67.230000, 37.180000, 37.075000, 36.995000, 36.780000, 36.625000, 36.340000, 36.165000, 65.965000, 95.420000, 95.305000, 65.305000, 35.005000, 64.980000, 64.945000, 34.910000, 34.625000, 64.510000, 34.390000, 63.600000, 33.270000, 33.075000, 63.045000, 62.980000, 32.915000, 62.865000, 62.700000, 2.530000, 92.500000, 62.370000, 62.310000, 92.140000, 62.125000, 32.110000, 32.025000, 91.925000, 61.580000, 61.575000, 91.355000, 31.170000, 120.865000, 90.390000, 90.260000, 59.850000, 89.740000, 59.725000, 29.240000, 119.160000, 89.090000, 118.905000, 58.780000, 58.360000, 58.355000, 0.100000, 27.770000, 27.455000, 87.435000, 117.235000, 27.200000, 27.015000, 26.985000, 56.840000, 56.815000, 86.615000, 86.455000, 25.575000, 85.525000, 55.040000, 54.980000, 24.800000, 54.040000, 24.005000, 54.000000, 53.960000, 53.895000, 53.785000, 23.590000, 83.440000, 53.395000, 23.390000, 53.235000, 83.140000, 23.100000, 53.065000, 22.900000, 52.715000, 52.705000, 112.435000, 22.385000, 52.325000, 52.125000, 52.040000, 51.980000, 21.870000, 111.390000, 81.380000, 80.980000, 50.805000, 50.565000, 80.335000, 19.730000, 109.430000, 49.235000, 79.130000, 79.035000, 108.980000, 18.385000, 48.315000, 18.295000, 78.270000, 77.955000, 77.940000, 77.790000, 77.735000, 47.430000, 47.285000, 47.160000, 77.025000, 76.990000, 16.665000, 16.515000, 46.365000, 76.045000, 45.965000, 75.855000, 15.680000, 45.620000, 45.505000, 45.500000, 15.175000, 74.805000, 44.780000, 44.715000, 44.550000, 74.525000, 14.470000, 44.335000, 44.275000, 73.805000, 103.750000, 43.645000, 43.200000, 42.850000, 42.680000, 72.655000, 11.945000, 71.730000, 41.340000, 11.330000, 71.135000, 71.005000, 40.940000, 40.910000, 40.840000, 10.505000, 10.395000, 70.155000, 39.910000, 39.740000, 69.105000, 38.920000, 68.525000, 8.300000, 68.135000, 38.095000, 38.050000, 67.715000, 37.685000, 67.670000, 37.130000, 37.025000, 66.975000, 66.890000, 66.870000, 36.425000, 36.375000, 6.370000, 66.295000, 96.155000, 65.965000, 95.680000, 35.675000, 95.530000, 65.230000, 65.080000, 34.585000, 64.565000, 4.425000, 34.400000, 64.275000, 34.095000, 93.785000, 92.980000, 62.840000, 62.520000, 32.450000, 32.360000, 2.270000, 92.270000, 32.160000, 62.150000, 62.135000, 62.040000, 61.980000, 31.970000, 31.870000, 61.545000, 31.540000, 31.340000, 61.295000, 30.900000, 90.765000, 30.535000, 30.305000, 60.215000, 30.160000, 29.935000, 59.880000, 29.850000, 89.750000, 59.740000, 29.715000, 29.565000, 29.525000, 59.335000, 29.325000, 59.280000, 29.015000, 58.925000, 58.570000, 58.415000, 28.355000, 28.290000, 58.210000, 57.805000, 57.620000, 57.340000, 26.390000, 26.205000, 55.975000, 25.945000, 25.900000, 25.875000, 55.630000, 85.560000, 115.490000, 55.435000, 25.360000, 55.130000, 25.085000, 85.085000, 54.995000, 54.840000, 0.100000, 84.610000, 54.555000, 84.535000, 84.410000, 53.980000, 23.700000, 53.470000, 53.400000, 52.565000, 82.550000, 52.305000, 52.085000, 21.835000, 51.575000, 81.575000, 21.545000, 51.300000, 21.180000, 51.070000, 51.070000, 20.830000, 0.100000, 20.445000, 110.375000, 80.235000, 20.150000, 49.295000, 79.180000, 18.985000, 0.100000, 108.805000, 18.590000, 48.505000, 48.380000, 18.365000, 48.355000, 48.260000, 78.245000, 47.880000, 47.545000, 47.445000, 16.880000, 76.870000, 76.840000, 76.695000, 46.635000, 46.510000, 45.890000, 15.775000, 75.675000, 75.575000, 45.565000, 75.535000, 105.450000, 75.320000, 15.315000, 45.040000, 104.760000, 44.740000, 44.145000, 44.135000, 74.120000, 14.090000, 73.975000, 43.680000, 43.300000, 73.225000, 73.130000, 12.775000, 12.330000, 72.090000, 42.050000, 41.760000, 71.650000, 11.445000, 11.425000, 11.415000, 71.325000, 41.185000, 41.005000, 40.750000, 70.625000, 40.545000, 70.475000, 40.380000, 40.070000, 10.060000, 39.510000, 39.320000, 69.310000, 68.940000, 68.920000, 68.360000, 68.320000, 38.135000, 37.755000, 37.705000, 37.630000, 37.525000, 37.175000, 36.960000, 36.905000, 36.895000, 36.535000, 36.360000, 66.095000, 65.565000, 35.505000, 65.245000, 35.005000, 34.860000, 34.770000, 94.585000, 64.550000, 64.415000, 64.355000, 64.210000, 63.755000, 63.715000, 63.640000, 33.550000, 33.070000, 92.875000, 32.800000, 62.525000, 92.210000, 2.185000, 62.055000, 2.005000, 31.965000, 61.945000, 61.130000, 91.115000, 31.010000, 60.710000, 90.620000, 30.490000, 30.345000, 60.070000, 60.045000, 0.000000, 29.940000, 59.870000, 29.505000, 59.260000, 59.205000, 89.015000, 58.935000, 88.885000, 58.805000, 58.775000, 58.515000, 88.380000, 28.150000, 0.100000, 57.880000, 27.280000, 57.255000, 57.255000, 57.110000, 56.970000, 26.850000, 56.830000, 26.675000, 56.650000, 56.585000, 26.515000, 26.315000, 56.300000, 56.135000, 25.895000, 85.820000, 25.585000, 85.205000, 24.765000, 54.570000, 54.490000, 54.260000, 84.020000, 24.015000, 24.015000, 23.775000, 53.760000, 23.165000, 52.740000, 52.735000, 22.490000, 52.210000, 52.040000, 22.000000, 51.880000, 21.700000, 81.610000, 51.470000, 21.420000, 81.390000, 21.365000, 80.985000, 80.830000, 50.800000, 50.345000, 80.280000, 80.050000, 19.890000, 49.820000, 79.390000, 49.360000, 19.150000, 48.805000, 78.650000, 48.495000, 18.100000, 18.025000, 77.965000, 17.835000, 17.825000, 47.340000, 17.285000, 47.195000, 77.040000, 16.935000, 0.100000, 76.560000, 76.495000, 46.305000, 46.285000, 45.955000, 45.690000, 75.610000, 0.100000, 75.450000, 45.410000, 45.145000, 14.965000, 44.755000, 44.650000, 44.545000, 44.490000, 44.450000, 74.090000, 44.035000, 73.895000, 73.640000, 43.440000, 73.305000, 42.875000, 12.785000, 12.670000, 72.375000, 42.300000, 72.295000, 72.290000, 72.280000, 42.140000, 42.135000, 41.575000, 41.570000, 71.520000, 11.310000, 40.985000, 10.840000, 40.640000, 10.530000, 40.490000, 70.370000, 0.100000, 40.155000, 10.115000, 40.090000, 39.795000, 9.605000, 69.580000, 8.985000, 68.645000, 68.640000, 38.070000, 67.965000, 67.795000, 7.675000, 37.665000, 67.535000, 37.370000, 37.240000, 37.195000, 36.735000, 36.460000, 36.385000, 66.360000, 36.185000, 66.165000, 36.080000, 65.975000, 65.775000, 65.525000, 34.920000, 64.840000, 34.280000, 34.245000, 33.815000, 63.685000, 33.150000, 33.135000, 62.965000, 2.470000, 32.460000, 32.360000, 62.270000, 32.125000, 2.025000, 61.870000, 31.440000, 31.325000, 1.245000, 1.005000, 30.840000, 30.710000, 60.700000, 60.225000, 60.170000, 30.095000, 0.100000, 29.775000, 0.100000, 59.580000, 29.365000, 29.325000, 58.930000, 28.915000, 28.895000, 58.700000, 58.610000, 0.100000, 27.690000, 57.645000, 0.100000, 27.045000, 56.810000, 56.705000, 56.660000, 26.510000, 56.440000, 56.370000, 56.325000, 26.045000, 0.100000, 25.600000, 55.540000, 55.500000, 25.350000, 25.225000, 25.120000, 55.085000, 25.080000, 55.075000, 0.100000, 54.655000, 24.625000, 54.460000, 24.350000, 53.715000, 53.665000, 53.645000, 23.370000, 23.285000, 53.035000, 22.760000, 52.740000, 52.715000, 21.760000, 51.590000, 51.355000, 51.165000, 50.985000, 50.910000, 20.560000, 50.500000, 0.100000, 50.165000, 0.100000, 19.960000, 49.955000, 49.555000, 49.440000, 19.020000, 18.880000, 18.270000, 48.255000, 18.185000, 48.120000, 48.015000, 17.530000, 47.095000, 46.845000, 16.650000, 0.100000, 46.470000, 16.085000, 46.040000, 45.975000, 45.940000, 15.890000, 0.100000, 45.225000, 15.125000, 45.075000, 44.990000, 14.890000, 44.740000, 44.665000, 44.610000, 14.360000, 43.960000, 43.825000, 43.485000, 43.215000, 12.950000, 42.870000, 12.700000, 12.690000, 42.585000, 42.450000, 12.205000, 42.085000, 42.080000, 41.900000, 41.900000, 0.100000, 41.690000, 41.235000, 41.165000, 10.670000, 9.375000, 0.100000, 39.080000, 39.055000, 39.025000, 38.900000, 38.695000, 38.670000, 8.525000, 38.160000, 8.045000, 37.650000, 6.765000, 35.990000, 35.735000, 35.470000, 35.245000, 35.225000, 35.045000, 33.935000, 2.435000, 32.245000, 32.185000, 1.820000, 0.900000, 30.650000, 0.100000, 29.920000, 29.890000, 29.865000, 29.700000, 29.530000, 29.450000, 29.135000, 28.480000, 27.340000, 27.325000, 26.590000, 25.335000, 0.100000, 24.510000, 24.465000, 0.100000, 0.100000, 20.970000, 19.050000, 17.475000, 17.190000, 17.080000, 16.610000, 0.100000, 15.885000, 15.835000, 0.100000, ]

plt.hist(data, bins=200)
plt.show()