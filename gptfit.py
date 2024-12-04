import numpy as np

# Define the Gaussian function (PDF)
def gaussian(x, mu, sigma, weight):
    return weight / (np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Parameters for A(x), B(x), C(x)
params = [
    {"mu": 15, "sigma": 5, "weight": 730},   # A(x)
    {"mu": 44, "sigma": 5, "weight": 330},   # B(x)
    {"mu": 68, "sigma": 2.5, "weight": 50}   # C(x)
]

# Function to compute T(x)
def T(x):
    return sum(gaussian(x, p["mu"], p["sigma"], p["weight"]) for p in params)

# Total area (integral of T(x)) - the sum of the weights
total_area = sum(p["weight"] for p in params)  # This is 1110

# Function to compute P(x), the normalized probability distribution
def P(x):
    return T(x) / total_area

# Function to compute the CDF F(x) numerically (Riemann sum)
def F(x, num_points=1000):
    # We calculate the integral using a Riemann sum from -infinity to x
    x_values = np.linspace(-100, x, num_points)
    dx = x_values[1] - x_values[0]
    return np.sum(P(x_values) * dx)

# Bisection method to find the quantile function Q(p)
def bisection_method(p, lower=-100, upper=100, tol=1e-6, max_iter=1000):
    # Check initial conditions
    F_lower = F(lower)
    F_upper = F(upper)
    
    if F_lower > p or F_upper < p:
        raise ValueError(f"Probability p={p} is out of bounds for the CDF.")
    
    # Bisection loop
    for _ in range(max_iter):
        mid = (lower + upper) / 2
        F_mid = F(mid)
        
        # Check if the midpoint is close enough to the target probability
        if abs(F_mid - p) < tol:
            return mid
        
        # Narrow down the search interval
        if F_mid < p:
            lower = mid
        else:
            upper = mid
    
    raise ValueError("Bisection method did not converge.")

# Example: Find the quantile corresponding to a cumulative probability p = 0.95
p = 0.2
quantile_value = bisection_method(p)
print(f"Quantile corresponding to p = {p}: {quantile_value}")