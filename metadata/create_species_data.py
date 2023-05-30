import os
import shutil

import pandas as pd
from tqdm import tqdm

import config

destination_folder = "dataset"
os.makedirs(destination_folder, exist_ok=True)
print("Create Data Folders Please Wait...\n")

meta_data_file = config.META_DATA_FILE
audio_directory = "metadata/wavfiles"
metadata_df = pd.read_csv(meta_data_file)

for index,row in tqdm(metadata_df.iterrows(),total = len(metadata_df)):
    species = row[config.TARGET_COL_NAME]
    audio_file = row['filename']
    species_folder = os.path.join(destination_folder, species)
    
    if not os.path.exists(species_folder):
        os.makedirs(species_folder)
    source_path = os.path.join(audio_directory, audio_file)
    destination_path = os.path.join(species_folder, os.path.basename(audio_file))    
    shutil.copy(source_path, destination_path)
    
print(f"Created Each Bird Name Folders @ 'dataset/' Folder {destination_folder}")


