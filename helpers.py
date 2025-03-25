"""
Script to fetch the DT online histograms and turn them into parquet files.
This has to be done as the DQM folks were storing the 2022 data and the usual scripts dont work anymore!
"""

import numpy as np
import pandas as pd
import uproot
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

def process_run(args):
    """
    Process a run either in online or offline mode.

    Parameters:
        args (tuple): A tuple containing:
            - run (str): Run identifier.
            - path_to_DT_files (str): Base path to the DT ROOT files.
            - histogram_type (str): Substring to filter the histograms.
            - path_to_histograms_inside_root_files (str): Subdirectory path inside the ROOT file where histograms are stored.
            - is_online (bool): Flag indicating whether to use online mode (True) or offline mode (False).

    Returns:
        list: A list of dictionaries with keys 'Name', 'Run', and 'Array'.
    """
    run, path_to_DT_files, histogram_type, path_to_histograms_inside_root_files, is_online = args
    new_rows = []
    try:
        if is_online:
            # Online mode: open the ROOT file directly
            file_path = os.path.join(path_to_DT_files, f"DQM_V0001_DT_R000{run}.root")
            file = uproot.open(file_path)
            path_to_histos = f"DQMData/Run {run}/DT/Run summary/{path_to_histograms_inside_root_files}/"
            histos = file[path_to_histos].keys()
            for histo in histos:
                histo_name = histo.split('/')[-1]
                if histogram_type in histo and histo_name:
                    array = np.array(file[path_to_histos + histo].values())
                    new_rows.append({'Name': histo_name, 'Run': run, 'Array': array})
        else:
            # Offline mode: find the file in a folder structure based on the run number
            folder_prefix = f"000{str(run)[:4]}xx"
            folder_path = os.path.join(path_to_DT_files, folder_prefix)
            
            if not os.path.isdir(folder_path):
                print( folder_path )
                print(f"Folder {folder_prefix} not found for run {run}. ---- ")
            else:
                found = False
                for file_name in os.listdir(folder_path):
                    if f"R000{run}" in file_name and file_name.endswith(".root"):
                        file_path = os.path.join(folder_path, file_name)
                        file = uproot.open(file_path)
                        path_to_histos = f"DQMData/Run {run}/DT/Run summary/{path_to_histograms_inside_root_files}/"
                        histos = file[path_to_histos].keys()
                        for histo in histos:
                            histo_name = histo.split('/')[-1]
                            # Ensure the histogram name is valid, contains histogram_type, and avoid those with 'Task'
                            if str(histogram_type) in histo and histo_name and 'Task' not in histo:
                                array = np.array(file[path_to_histos + histo].values())
                                new_rows.append({'Name': histo_name, 'Run': run, 'Array': array})
                        found = True
                        break
                if not found:
                    print(f"ROOT file not found for run {run} in folder {folder_prefix}.")
    except Exception as e:
        print(f"Error processing run {run}: {e}")
    return new_rows

# Helper function for parallel execution
def process_run__(args):
    run, path_to_DT_files, histogram_type, path_to_histograms_inside_root_files = args
    IsOnline = True
    new_rows = []
    try:
        if IsOnline:
            file = uproot.open(f"{path_to_DT_files}DQM_V0001_DT_R000{run}.root")
            path_to_histos = f"DQMData/Run {run}/DT/Run summary/{path_to_histograms_inside_root_files}/"
            histos = file[path_to_histos].keys()
            for histo in histos:
                if histogram_type in histo and histo.split('/')[-1]:
                    name = histo.split('/')[-1]
                    array = np.array(file[path_to_histos + histo].values())
                    new_rows.append({'Name': name, 'Run': run, 'Array': array})
        else:

            # Iterate through folders and match runs
            for run in run_list:
                folder_prefix = f"000{run[:4]}xx"
                folder_path = os.path.join(path_to_DT_files, folder_prefix)

                # Check if the folder exists
                if not os.path.isdir(folder_path):
                    print(f"Folder {folder_prefix} not found.")
                    continue
                
                # Search for the ROOT file inside the matched folder
                for file_name in os.listdir(folder_path):
                    if f"R000{run}" in file_name and file_name.endswith(".root"):
                        file_path = os.path.join(folder_path, file_name)

                        file = uproot.open(file_path)
                        path_to_histos = f"DQMData/Run {run}/DT/Run summary/{path_to_histograms_inside_root_files}/"
                        
                        histos = file[path_to_histos].keys()
                        
                        for histo in histos:
                            
                            # Task is not used as the num and denum are there ...
                            if str(histogram_type) in histo and histo.split('/')[-1] and 'Task' not in histo:
                                
                                name = histo.split('/')[-1]
                                array = file[path_to_histos + histo].values()  
                                new_rows.append({'Name': name, 'Run': run, 'Array': array})

    except Exception as e:
        print(f"Error processing run {run}: {e}")
    return new_rows

# Main parallelized function
def read_Occupancy_histograms_parallel(path_to_root_files, histogram_type, path_to_histograms_inside_root_files ,run_list, max_workers=24):
    tasks = [(run, path_to_root_files, histogram_type, path_to_histograms_inside_root_files, False) for run in run_list]

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

### this only works for offline histograms, as online histograms are zipped in a weird format
def read_histograms_in_EOS(path_to_root_files, path_to_histograms_inside_root_files, histogram_type, run_list, IsOnline = False):
      
    path_to_DT_files = path_to_root_files
    path_inside_root = path_to_histograms_inside_root_files

    # for debuggin only!
    run_list = run_list[20:30]

    # for debuggin only!
    #run_list = run_list[:15]

    Occupancy_histograms = []
    TimeBox_histograms   = []

    # Lets create a pandas dataframe to store the histograms and names
    Occupancy_df = pd.DataFrame(columns=['Name', 'Run', 'Array'])

    new_rows = []
    counter = 0
    # Iterate through folders and match runs
    for run in run_list:
        folder_prefix = f"000{run[:4]}xx"
        folder_path = os.path.join(path_to_DT_files, folder_prefix)

        # Check if the folder exists
        if not os.path.isdir(folder_path):
            print(f"Folder {folder_prefix} not found.")
            continue
        
        # Search for the ROOT file inside the matched folder
        for file_name in os.listdir(folder_path):
            if f"R000{run}" in file_name and file_name.endswith(".root"):
                file_path = os.path.join(folder_path, file_name)

                file = uproot.open(file_path)
                path_to_histos = f"DQMData/Run {run}/DT/Run summary/{path_inside_root}/"
                
                histos = file[path_to_histos].keys()
                
                for histo in histos:
                    
                    # Task is not used as the num and denum are there ...
                    if str(histogram_type) in histo and histo.split('/')[-1] and 'Task' not in histo:
                        
                        name = histo.split('/')[-1]
                        array = file[path_to_histos + histo].values()  
                        new_rows.append({'Name': name, 'Run': run, 'Array': array})
                        counter += 1
                          
    # Append the collected rows to the DataFrame
    Occupancy_df = append_to_dataframe(Occupancy_df, new_rows) 

    names = Occupancy_df.Name
    runs  = Occupancy_df.Run
        
    # Lets save the dataframe
                
    return Occupancy_df

def read_histograms_in_EOS_eff(path_to_root_files, path_to_histograms_inside_root_files, histogram_type, run_list):
      
    path_to_DT_files = path_to_root_files
    path_inside_root = path_to_histograms_inside_root_files

    # for debuggin only!
    run_list = run_list[20:40]

    # for debuggin only!
    #run_list = run_list[:15]

    Occupancy_histograms = []
    TimeBox_histograms   = []

    # Lets create a pandas dataframe to store the histograms and names
    Occupancy_df = pd.DataFrame(columns=['Name', 'Run', 'Array'])

    new_rows = []
    counter = 0
    # Iterate through folders and match runs
    for run in run_list:
        folder_prefix = f"000{run[:4]}xx"
        folder_path = os.path.join(path_to_DT_files, folder_prefix)

        # Check if the folder exists
        if not os.path.isdir(folder_path):
            print(f"Folder {folder_prefix} not found.")
            continue
        
        # Search for the ROOT file inside the matched folder
        for file_name in os.listdir(folder_path):
            if f"R000{run}" in file_name and file_name.endswith(".root"):
                file_path = os.path.join(folder_path, file_name)

                file = uproot.open(file_path)
                path_to_histos = f"DQMData/Run {run}/DT/Run summary/03-LocalTrigger-TM/"
                path_to_histos_num = f"DQMData/Run {run}/DT/Run summary/03-LocalTrigger-TM/Task/"
                path_to_histos_denum = f"DQMData/Run {run}/DT/Run summary/03-LocalTrigger-TM/Task/"
                
                histos       = file[path_to_histos].keys()
                histos_num   = file[path_to_histos_num].keys()
                histos_denum = file[path_to_histos_denum].keys()
                
                for histo in histos:
                    
                    # Task is not used as the num and denum are there ...
                    if str(histogram_type) in histo and histo.split('/')[-1] and 'Task' not in histo:
                        
                        for histo_num in histos_num:
                            if histo[-4:-2] in histo_num and "Num" in histo_num:
                                array_num = np.array(file[path_to_histos_num + histo_num].values())
                    
                        ## Now for the denominator
                        for histo_denum in histos_denum:
                            if histo[-4:-2] in histo_denum and "Denum" in histo_denum:
                                array_denum = np.array(file[path_to_histos_denum + histo_denum].values())
                        
                        name = histo.split('/')[-1]
                        array = file[path_to_histos + histo].values()  
                        new_rows.append({'Name': name, 'Run': run, 'Array': array, 'Array_num': array_num, 'Array_denum': array_denum})
                        counter += 1
                          
    # Append the collected rows to the DataFrame
    Occupancy_df = append_to_dataframe(Occupancy_df, new_rows) 

    names = Occupancy_df.Name
    runs  = Occupancy_df.Run
        
    # Lets save the dataframe
                
    return Occupancy_df