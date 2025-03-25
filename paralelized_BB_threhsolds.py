import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
import os
import csv
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

### other scripts
import helpers
import statistical_tools.beta_binomial_test as bb_test

# Global variables for multiprocessing
global_df_dict = {}

def initialize_global_dataframe(dataframe):
    global global_df_dict
    global_df_dict = dataframe.set_index(["Name", "Run"])['Array'].to_dict()

def compute_metrics(args):
    histo, run, reference_runs = args
    
    global global_df_dict
    try:
        histo_data = global_df_dict.get((histo, str(run)))
        list_of_reference_hists = [global_df_dict.get((histo, str(ref_run))) for ref_run in reference_runs]

        Chi2, maxpull = bb_test.beta_binomial(histo_data, list_of_reference_hists)
        return Chi2, abs(maxpull)

    except Exception as e:
        print(f"Error computing metrics for histo={histo}, run={run}: {e}")
        return None  # To indicate a problem

def main():
    if len(sys.argv) < 2:
        print("Usage: python setting_BB_thresholds.py <path_to_config>")
        sys.exit(1)

    config = json.load(open(sys.argv[1]))

    histogram_type = config['histogram_type']

    path_to_histograms_inside_root_files = config['path_to_histograms_inside_root_files']

    #### reading the dataframe and histograms from inside the .root files
    dataframe  = helpers.read_Occupancy_histograms_parallel(config['path_to_root_files'], config['histogram_type'], config['path_to_histograms_inside_root_files'], config['good_run_list'], max_workers=24)
    #dataframe = pd.read_pickle("occupancy_histograms.pkl")
    
    initialize_global_dataframe(dataframe)
    number_of_reference_runs_ = config["number_of_reference_runs"]

    path_to_outputs = config['path_to_outputs']
    if os.path.exists(path_to_outputs):
        os.remove(path_to_outputs)

    # Get a sorted list of runs from your dataframe
    runs = sorted(dataframe.Run.unique())
    histograms = dataframe.Name.unique()

    for k in number_of_reference_runs_:
        # Select data runs: only those runs with at least k runs following them.
        data_runs = runs[:-k]
        # For each data run, pick the next k runs as its reference runs.
        reference_runs_list = [runs[i+1:i+1+k] for i in range(len(runs) - k)]
        
        Chi_metrics = []
        Maxpull_metrics = []

        # Create tasks for each histogram and each (data_run, reference_runs) pair.
        tasks = [
            (histo, data_run, ref_runs)
            for histo in histograms
            for data_run, ref_runs in zip(data_runs, reference_runs_list)
        ]

        with ProcessPoolExecutor(max_workers=24) as executor:
            results = list(tqdm(
                executor.map(compute_metrics, tasks),
                total=len(tasks),
                desc=f"Computing metrics for k={k}"
            ))
            
            for res in results:
                if res is not None:
                    chi, maxpull = res
                    Chi_metrics.append(chi)
                    Maxpull_metrics.append(maxpull)
                else:
                    print("Warning: A computation returned None, skipping...")

        # Compute statistical metrics for Chi2 and Maxpull values.
        chi2_mean   = np.mean(Chi_metrics)
        chi2_q80    = np.quantile(Chi_metrics, 0.80)
        chi2_q90    = np.quantile(Chi_metrics, 0.90)
        chi2_q95    = np.quantile(Chi_metrics, 0.95)
        chi2_q98    = np.quantile(Chi_metrics, 0.98)
        chi2_q99    = np.quantile(Chi_metrics, 0.99)
        
        maxpull_mean  = np.mean(Maxpull_metrics)
        maxpull_q80   = np.quantile(Maxpull_metrics, 0.80)
        maxpull_q90   = np.quantile(Maxpull_metrics, 0.90)
        maxpull_q95   = np.quantile(Maxpull_metrics, 0.95)
        maxpull_q98   = np.quantile(Maxpull_metrics, 0.98)
        maxpull_q99   = np.quantile(Maxpull_metrics, 0.99)

        header = [
            "nReference_runs", "Chi2_mean", "Chi2_80th_quantile", "Chi2_90th_quantile",
            "Chi2_95th_quantile", "Chi2_98th_quantile", "Chi2_99th_quantile", "Maxpull_mean",
            "Maxpull_80th_quantile", "Maxpull_90th_quantile", "Maxpull_95th_quantile",
            "Maxpull_98th_quantile", "Maxpull_99th_quantile"
        ]

        file_exists = os.path.isfile(path_to_outputs)
        write_header = not file_exists or os.stat(path_to_outputs).st_size == 0

        with open(path_to_outputs, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            row = [
                k,
                f"{chi2_mean:.4f}", f"{chi2_q80:.4f}", f"{chi2_q90:.4f}",
                f"{chi2_q95:.4f}", f"{chi2_q98:.4f}", f"{chi2_q99:.4f}",
                f"{maxpull_mean:.4f}", f"{maxpull_q80:.4f}", f"{maxpull_q90:.4f}",
                f"{maxpull_q95:.4f}", f"{maxpull_q98:.4f}", f"{maxpull_q99:.4f}"
            ]
            writer.writerow(row)

        folder = os.path.dirname(path_to_outputs) or "."

        # Plot and save the Chi2 histogram.
        plt.figure()
        bins = np.linspace(0, chi2_q98, 40)
        plt.hist(Chi_metrics, bins=bins, edgecolor='black')
        plt.title("Chi2 Histogram")
        plt.xlabel("Chi2 values")
        plt.ylabel("Frequency")
        plt.xlim(0, chi2_q98)
        chi2_text = f"Mean: {chi2_mean:.4f}\n90th quantile: {chi2_q90:.4f}\n95th quantile: {chi2_q95:.4f}"
        plt.text(0.95 * chi2_q98, plt.ylim()[1]*0.8, chi2_text, ha='right', 
                bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig(os.path.join(folder, f'{histogram_type}_chi2_histogram_{k}.png'))
        plt.close()

        # Plot and save the Maxpull histogram.
        plt.figure()
        bins = np.linspace(0, maxpull_q98, 40)
        plt.hist(Maxpull_metrics, bins=bins, edgecolor='black')
        plt.title("Maxpull Histogram")
        plt.xlabel("Maxpull values")
        plt.ylabel("Frequency")
        plt.xlim(0, maxpull_q98)
        maxpull_text = f"Mean: {maxpull_mean:.4f}\n90th quantile: {maxpull_q90:.4f}\n95th quantile: {maxpull_q95:.4f}"
        plt.text(0.95 * maxpull_q98, plt.ylim()[1]*0.8, maxpull_text, ha='right', 
                bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig(os.path.join(folder, f'{histogram_type}_maxpull_histogram_{k}.png'))
        plt.close()


if __name__ == "__main__":
    main()
