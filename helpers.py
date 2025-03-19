"""
Script to fetch the DT online histograms and turn them into parquet files.
This has to be done as the DQM folks were storing the 2022 data and the usual scripts dont work anymore!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import os
import uproot

# Function to append data
def append_to_dataframe(dataframe, new_rows):
    return pd.concat([dataframe, pd.DataFrame(new_rows)], ignore_index=True)

def read_Occupancy_histograms(path_to_root_files, path_to_histograms_inside_root_files, histogram_type, run_list):
      
    path_to_DT_files = path_to_root_files
    path_inside_root = path_to_histograms_inside_root_files

    #run_list = run_list[:10]
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
