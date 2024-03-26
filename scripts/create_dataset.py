""" Module for creating the dataset by combining categorical/numerical and text csv. files.

First, download categorical/numerical data into data folder as described in README.md, then:

Run in CLI example:
    'python create_dataset.py'

"""

import yaml
import os
import pandas as pd
#from pathlib import Path


    
def create_data():
    
    # Get the current user directory (the directory where the script is located)
    current_directory = os.getcwd()
    # Define the file name or file path relative to the current directory
    file_name = "Churn_Model.csv"
        
    # Construct the full file path by joining the current directory and file name
    file_path = os.path.join(current_directory, "data")
    params_path = os.path.join(file_path, file_name)
    
    print(params_path)
    try:
         # save data
        train_data = pd.read_csv(params_path)
        print(f"Train dataset shape: {train_data.shape}")
    except FileNotFoundError:
        print(f"The file '{file_name}' was not found in the current directory.")
    except IOError as e:
        print(f"An error occurred while trying to read the file: {e}")
        
    return train_data
    
    
    