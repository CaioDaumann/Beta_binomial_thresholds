import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import json

if len(sys.argv) < 2:
    print("Usage: python fit_thresholds.py <path_to_output_file>")
    exit()

### reading the json config file
config = json.load(open(sys.argv[1]))

path_to_outputs = config['path_to_outputs']
histogram_type = config['histogram_type']

# New function to fit
def custom_fit(x, a, b):
    return a / x + b**2

# Load the data
output_file = path_to_outputs  # Replace this with your actual file path
df = pd.read_csv(output_file)

# Add an additional point for 1 reference run (manually adjust if needed)
"""
additional_point = pd.DataFrame({
    'nReference_runs': [1],
    'Chi2_95th_quantile': [df['Chi2_90th_quantile'].iloc[0]],  # Example value; adjust if you have specific data
    'Maxpull_95th_quantile': [df['Maxpull_90th_quantile'].iloc[0]],  # Example value; adjust if you have specific data
})
df = pd.concat([additional_point, df], ignore_index=True).sort_values(by='nReference_runs')
"""

# Extract necessary columns
x_data = df['nReference_runs'].values
chi2_q95 = df['Chi2_98th_quantile'].values
maxpull_q95 = df['Maxpull_98th_quantile'].values

# Define uncertainties based on reference run counts
unc_dict = {1: 0.2, 2: 0.15, 4: 0.125, 6: 0.1, 8: 0.075}
y_err_chi2 = np.array([chi2_q95[i] * unc_dict.get(x, 0.05) for i, x in enumerate(x_data)])
y_err_maxpull = np.array([maxpull_q95[i] * unc_dict.get(x, 0.05) for i, x in enumerate(x_data)])

# Bounds to constrain b >= 1
bounds = ([0, 1.0], [np.inf, np.inf])

# Fit for Chi2
params_chi2, cov_chi2 = curve_fit(custom_fit, x_data, chi2_q95, sigma=y_err_chi2, bounds=bounds)

# Fit for Maxpull
params_maxpull, cov_maxpull = curve_fit(custom_fit, x_data, maxpull_q95, sigma=y_err_maxpull, bounds=bounds)

# Plotting the fits
x_fit = np.linspace(min(x_data), max(x_data), 100)

plt.figure(figsize=(12, 5))

# Chi2 plot
plt.subplot(1, 2, 1)
plt.errorbar(x_data, chi2_q95, yerr=y_err_chi2, fmt='o', label='Chi2 Data (95th quantile)', capsize=4)
plt.plot(x_fit, custom_fit(x_fit, *params_chi2), 'r--', label=f'Fit: {params_chi2[0]:.4f}/x + {params_chi2[1]:.4f}²')
plt.xlabel('Number of Reference Runs', fontsize=16)
plt.ylabel('Chi2 95th Quantile', fontsize=16)
plt.title(f'{histogram_type} - Chi2 Fit', fontsize=16)
plt.legend()

# Maxpull plot
plt.subplot(1, 2, 2)
plt.errorbar(x_data, maxpull_q95, yerr=y_err_maxpull, fmt='o', label='Maxpull Data (95th quantile)', capsize=4)
plt.plot(x_fit, custom_fit(x_fit, *params_maxpull), 'r--', label=f'Fit: {params_maxpull[0]:.4f}/x + {params_maxpull[1]:.4f}²')
plt.xlabel('Number of Reference Runs', fontsize=16)
plt.ylabel('Maxpull 95th Quantile', fontsize=16)
plt.title(f'{histogram_type} - Maxpull Fit', fontsize=16)
plt.legend()

plt.tight_layout()
plt.savefig(f'fits_outputs/{histogram_type}fits_quantiles.png')
plt.show()

# Printing results
print("Chi2 fit parameters: a = {:.4f}, b = {:.4f}".format(*params_chi2))
print("Maxpull fit parameters: a = {:.4f}, b = {:.4f}".format(*params_maxpull))