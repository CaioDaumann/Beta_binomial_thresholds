import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json
import os
import csv

### other scripts
import helpers
import statistical_tools.beta_binomial_test as bb_test

def main():
    
    if len(sys.argv) < 1:
        print("Usage: python setting_BB_thresholds.py <path_to_config>")
        sys.exit(1)
        
    config = json.load(open(sys.argv[1]))

    ########### comments first two if .pkl is already there ...
    dataframe = helpers.read_Occupancy_histograms(config['path_to_root_files'], config['path_to_histograms_inside_root_files'], config['histogram_type'], config['good_run_list'])
    dataframe.to_pickle("occupancy_histograms.pkl")
    #dataframe = pd.read_pickle("occupancy_histograms.pkl")
    
    number_of_reference_runs_ = config[ "number_of_reference_runs"]
    
    # opening and creating a .txt file to store the threhsolds information
    path_to_outputs = config['path_to_outputs']

    # Lets remove the .txt if it exists in order not to cluster things ...
    try:
        os.remove(path_to_outputs)
    except:
        pass
    
    for number_of_reference_runs in number_of_reference_runs_:
     
        reference_runs = pd.Series(dataframe.Run.unique()).sample(number_of_reference_runs, random_state=42).values

        # Convert the unique runs to a list, then remove the reference run if present.
        data_runs = list(dataframe.Run.unique())
        histograms = dataframe.Name.unique() 
        
        # Lets remove the reference runs from the data runs
        data_runs = [run for run in data_runs if run not in reference_runs]

        Chi_metrics     = []
        Maxpull_metrics = []
        for histo in histograms:
            for run in data_runs:
                
                histo_data = dataframe.loc[
                    (dataframe["Name"] == histo) & 
                    (dataframe["Run"] == str(run)), 
                    "Array"
                ].values[0]
                
                list_of_reference_hists = []
                for reference_run in reference_runs:
                
                    list_of_reference_hists.append(dataframe.loc[
                        (dataframe["Name"] == histo) & 
                        (dataframe["Run"] == str(reference_run)), 
                        "Array"
                    ].values[0])
                
                # sometimes the beta_binomial function fails, so we need to catch the exception
                try:
                    Chi2, maxpull = bb_test.beta_binomial(histo_data , list_of_reference_hists )
                    Chi_metrics.append(Chi2)
                    Maxpull_metrics.append(abs(maxpull))
                except:
                    pass

        # Plot histogram for Chi_metrics
        plt.figure()
        plt.hist(Chi_metrics, bins=30, edgecolor='black')
        plt.title('Histogram of Chi Metrics')
        plt.xlabel('Chi2 value')
        plt.ylabel('Frequency')
        plt.savefig('plots/Chi2_metrics.png')

        # Plot histogram for Maxpull_metrics
        plt.figure()
        plt.hist(Maxpull_metrics, bins=30, edgecolor='black')
        plt.title('Histogram of Maxpull Metrics')
        plt.xlabel('Maxpull value')
        plt.ylabel('Frequency')
        plt.savefig('plots/Maxpull_metrics.png')
            
        chi2_mean = np.mean(Chi_metrics)
        chi2_q90 = np.quantile(Chi_metrics, 0.90)
        chi2_q95 = np.quantile(Chi_metrics, 0.95)
        chi2_q98 = np.quantile(Chi_metrics, 0.96)

        # Calculated metrics for Maxpull (using mean instead of std for annotation)
        maxpull_mean = np.mean(Maxpull_metrics)
        maxpull_q90 = np.quantile(Maxpull_metrics, 0.90)
        maxpull_q95 = np.quantile(Maxpull_metrics, 0.95)
        maxpull_q98 = np.quantile(Maxpull_metrics, 0.96)
            
        # Define the header (column names)
        header = [
            "nReference_runs",
            "Chi2_mean",
            "Chi2_90th_quantile",
            "Chi2_95th_quantile",
            "Maxpull_mean",
            "Maxpull_90th_quantile",
            "Maxpull_95th_quantile"
        ]

        # Check if the output file already exists and is non-empty.
        file_exists = os.path.isfile(path_to_outputs)
        write_header = not file_exists or os.stat(path_to_outputs).st_size == 0

        # Open the file in append mode
        with open(path_to_outputs, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if needed
            if write_header:
                writer.writerow(header)
            
            # Prepare the row with the calculated metrics
            row = [
                number_of_reference_runs,
                f"{chi2_mean:.4f}",
                f"{chi2_q90:.4f}",
                f"{chi2_q95:.4f}",
                f"{maxpull_mean:.4f}",
                f"{maxpull_q90:.4f}",
                f"{maxpull_q95:.4f}"
            ]
            
            # Write the row to the CSV file
            writer.writerow(row)
            
            # Create a summary string for printing (optional)
            summary = " | ".join(f"{col}: {val}" for col, val in zip(header, row))
            print(summary)

        # Determine the folder where the .txt file is saved
        folder = os.path.dirname(path_to_outputs)
        if not folder:
            folder = "."
        
        # Lets also save the distribution of the metrics
        # Create and save histogram for Chi2 scores
        plt.figure()
        bins = np.linspace(0, chi2_q98, 40)
        plt.hist(Chi_metrics, bins=bins, edgecolor='black')
        plt.title("Chi2 Histogram")
        plt.xlabel("Chi2 values")
        plt.ylabel("Frequency")
        plt.xlim(0, chi2_q98)  # Set x-axis from 0 to the 98th quantile

        # Annotate the plot with the computed metrics
        chi2_text = (f"Mean: {chi2_mean:.4f}\n"
                    f"90th quantile: {chi2_q90:.4f}\n"
                    f"95th quantile: {chi2_q95:.4f}")
        plt.text(0.95 * chi2_q98, plt.ylim()[1]*0.8, chi2_text,
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

        chi2_plot_path = os.path.join(folder, f'chi2_histogram_{number_of_reference_runs}.png')
        plt.savefig(chi2_plot_path)
        plt.close()  # Close the figure to free memory

        # Create and save histogram for Maxpull scores
        plt.figure()
        bins = np.linspace(0, maxpull_q98, 40)
        plt.hist(Maxpull_metrics, bins=bins, edgecolor='black')
        plt.title("Maxpull Histogram")
        plt.xlabel("Maxpull values")
        plt.ylabel("Frequency")
        plt.xlim(0, maxpull_q98)  # Set x-axis from 0 to the 98th quantile

        # Annotate the plot with the computed metrics
        maxpull_text = (f"Mean: {maxpull_mean:.4f}\n"
                        f"90th quantile: {maxpull_q90:.4f}\n"
                        f"95th quantile: {maxpull_q95:.4f}")
        plt.text(0.95 * maxpull_q98, plt.ylim()[1]*0.8, maxpull_text,
                horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

        maxpull_plot_path = os.path.join(folder, f'maxpull_histogram_{number_of_reference_runs}.png')
        plt.savefig(maxpull_plot_path)
        plt.close()  # Close the figure
        
if __name__ == "__main__":
    main()