# Data
import numpy as np
import matplotlib.pyplot as plt
thresholds = np.arange(0.19, 0.32, 0.02)
precision = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
coverage = [0.85, 0.83, 0.80, 0.78, 0.75, 0.70, 0.65]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision, label="Precision", marker='o', color="blue")
plt.plot(thresholds, coverage, label="Category Coverage", marker='s', color="green")
plt.axvline(x=0.27, color='r', linestyle='--', label="Optimal Threshold")
plt.title("Product Category Coverage vs. Precision", fontsize=14)
plt.xlabel("Threshold Values", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()
