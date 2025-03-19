"""
Script to fetch the DT online histograms and turn them into parquet files.
This has to be done as the DQM folks were storing the 2022 data and the usual scripts dont work anymore!
"""

import numpy as np
import pandas as pd
import uproot
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Helper function for parallel execution
def process_run(args):
    run, path_to_DT_files, histogram_type = args
    new_rows = []
    try:
        file = uproot.open(f"{path_to_DT_files}DQM_V0001_DT_R000{run}.root")
        path_to_histos = f"DQMData/Run {run}/DT/Run summary/01-Digi/"
        histos = file[path_to_histos].keys()
        for histo in histos:
            if histogram_type in histo and histo.split('/')[-1]:
                name = histo.split('/')[-1]
                array = np.array(file[path_to_histos + histo].values())
                new_rows.append({'Name': name, 'Run': run, 'Array': array})
    except Exception as e:
        print(f"Error processing run {run}: {e}")
    return new_rows

# Main parallelized function
def read_Occupancy_histograms_parallel(path_to_root_files, histogram_type, run_list, max_workers=24):
    tasks = [(run, path_to_root_files, histogram_type) for run in run_list]

    Occupancy_df = pd.DataFrame(columns=['Name', 'Run', 'Array'])
    all_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_run, tasks), total=len(tasks), desc="Processing runs"))
        for rows in results:
            all_rows.extend(rows)

    Occupancy_df = pd.concat([Occupancy_df, pd.DataFrame(all_rows)], ignore_index=True)

    return Occupancy_df


# Function to append data
def append_to_dataframe(dataframe, new_rows):
    return pd.concat([dataframe, pd.DataFrame(new_rows)], ignore_index=True)

def read_Occupancy_histograms(path_to_root_files, path_to_histograms_inside_root_files, histogram_type, run_list):
      
    path_to_DT_files = path_to_root_files
    path_inside_root = path_to_histograms_inside_root_files

    run_list = run_list[:10]
    no_training_list = [ 
        
            "OccupancyAllHits_perCh_W-2_St2_Sec1",
            "OccupancyAllHits_perCh_W-2_St2_Sec4",
            "OccupancyAllHits_perCh_W-2_St2_Sec8",
            "OccupancyAllHits_perCh_W-2_St2_Sec10",
            "OccupancyAllHits_perCh_W-2_St1_Sec12",
            "OccupancyAllHits_perCh_W-2_St2_Sec12",
            "OccupancyAllHits_perCh_W-1_St1_Sec12",
            "OccupancyAllHits_perCh_W0_St2_Sec12",
            "OccupancyAllHits_perCh_W0_St3_Sec6",
            "OccupancyAllHits_perCh_W0_St2_Sec1",
            "OccupancyAllHits_perCh_W1_St1_Sec1",
            "OccupancyAllHits_perCh_W1_St2_Sec5",
            "OccupancyAllHits_perCh_W1_St2_Sec11",
            "OccupancyAllHits_perCh_W2_St1_Sec8",
            "OccupancyAllHits_perCh_W-2_St2_Sec8"
                                            
        ]

    # for debuggin only!
    #run_list = run_list[:15]

    Occupancy_histograms = []
    TimeBox_histograms   = []

    # Lets create a pandas dataframe to store the histograms and names
    Occupancy_df = pd.DataFrame(columns=['Name', 'Run', 'Array'])

    new_rows = []
    counter = 0
    for run in run_list:
        print(f"Working on run {run}")
        try:
            file = uproot.open(path_to_DT_files+f"DQM_V0001_DT_R000{run}.root")
            path_to_histos = f"DQMData/Run {run}/DT/Run summary/01-Digi/"
            histos = file[path_to_histos].keys()
            for histo in histos:
                if( str(histogram_type) in histo and histo.split('/')[-1] ):
                        name = histo.split('/')[-1]
                        array = np.array(file[path_to_histos + histo].values())
                        new_rows.append({'Name': name, 'Run': run, 'Array': array})
                        counter += 1
        except:
                pass
                    
    # Append the collected rows to the DataFrame
    Occupancy_df = append_to_dataframe(Occupancy_df, new_rows) 

    names = Occupancy_df.Name
    runs  = Occupancy_df.Run
        
    # Lets save the dataframe
                
    return Occupancy_df
