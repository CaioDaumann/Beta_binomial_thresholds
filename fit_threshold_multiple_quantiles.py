import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import json

if len(sys.argv) < 2:
    print("Usage: python fit_thresholds.py <path_to_output_file>")
    exit()

# Reading the JSON config file
config = json.load(open(sys.argv[1]))
path_to_outputs = config['path_to_outputs']
histogram_type = config['histogram_type']

# Define the custom fitting function
def custom_fit(x, a, b):
    return a / x + b**2

# Load the data
df = pd.read_csv(path_to_outputs)

# Get the x-axis data (number of reference runs)
x_data = df['nReference_runs'].values

# Define the quantiles to fit and plot (80, 90, 95, and 98)
quantiles = [80, 90, 95, 98, 99]

# Prepare color list (you can customize these colors)
colors = ['blue', 'orange', 'green', 'red', 'purple']

# Dictionaries to store fit parameters for printing later
fit_params_chi2 = {}
fit_params_maxpull = {}

# Uncertainty dictionary based on reference run counts
unc_dict = {1: 0.2, 2: 0.15, 4: 0.125, 6: 0.1, 8: 0.075}

# Create a smooth x range for plotting the fitted curves
x_fit = np.linspace(min(x_data), max(x_data), 100)

plt.figure(figsize=(12, 5))

# --- Chi2 Plot ---
plt.subplot(1, 2, 1)
for i, quant in enumerate(quantiles):
    col_name = f'Chi2_{quant}th_quantile'
    if col_name not in df.columns:
        print(f"Column {col_name} not found in data.")
        continue
    # Extract the quantile data for Chi2
    chi2_data = df[col_name].values
    # Calculate uncertainties; if a key is missing in unc_dict, use default 0.05
    y_err = np.array([chi2_data[j] * unc_dict.get(x, 0.05) for j, x in enumerate(x_data)])
    # Fit the data with the custom function, enforcing b >= 1
    params, cov = curve_fit(custom_fit, x_data, chi2_data, sigma=y_err, bounds=([0, 1.0], [np.inf, np.inf]))
    fit_params_chi2[quant] = params
    # Generate fitted curve
    fitted_curve = custom_fit(x_fit, *params)
    # Plot data points with error bars using the same color
    plt.errorbar(x_data, chi2_data, yerr=y_err, fmt='o', capsize=4,
                 color=colors[i], label=f'Chi2 Data ({quant}th)')
    # Plot the fitted curve with the same color
    plt.plot(x_fit, fitted_curve, '--', color=colors[i],
             label=f'Fit ({quant}th): {params[0]:.4f}/x + {params[1]:.4f}²')

plt.xlabel('Number of Reference Runs', fontsize=16)
plt.ylabel('Chi2 Quantile', fontsize=16)
plt.title(f'{histogram_type} - Chi2 Fit', fontsize=16)
plt.legend()

# --- Maxpull Plot ---
plt.subplot(1, 2, 2)
for i, quant in enumerate(quantiles):
    col_name = f'Maxpull_{quant}th_quantile'
    if col_name not in df.columns:
        print(f"Column {col_name} not found in data.")
        continue
    # Extract the quantile data for Maxpull
    maxpull_data = df[col_name].values
    # Calculate uncertainties
    y_err = np.array([maxpull_data[j] * unc_dict.get(x, 0.05) for j, x in enumerate(x_data)])
    # Fit the data with the custom function, enforcing b >= 1
    params, cov = curve_fit(custom_fit, x_data, maxpull_data, sigma=y_err, bounds=([0, 1.0], [np.inf, np.inf]))
    fit_params_maxpull[quant] = params
    # Generate fitted curve
    fitted_curve = custom_fit(x_fit, *params)
    # Plot data points with error bars using the same color
    plt.errorbar(x_data, maxpull_data, yerr=y_err, fmt='o', capsize=4,
                 color=colors[i], label=f'Maxpull Data ({quant}th)')
    # Plot the fitted curve with the same color
    plt.plot(x_fit, fitted_curve, '--', color=colors[i],
             label=f'Fit ({quant}th): {params[0]:.4f}/x + {params[1]:.4f}²')

plt.xlabel('Number of Reference Runs', fontsize=16)
plt.ylabel('Maxpull Quantile', fontsize=16)
plt.title(f'{histogram_type} - Maxpull Fit', fontsize=16)
plt.legend()

plt.tight_layout()
plt.savefig(f'fits_outputs/{histogram_type}_fits_quantiles.png')
plt.show()

# Print out the fit parameters for each quantile
print("Chi2 fit parameters:")
for quant, params in fit_params_chi2.items():
    print(f"  {quant}th quantile: a = {params[0]:.4f}, b = {params[1]:.4f}")

print("\nMaxpull fit parameters:")
for quant, params in fit_params_maxpull.items():
    print(f"  {quant}th quantile: a = {params[0]:.4f}, b = {params[1]:.4f}")
