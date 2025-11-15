# preprocess.py
import glob
import os
import pandas as pd

EHR_DATA_FOLDER = '/Users/yashbishnoi/Downloads/Dundee university/90Days_Machine_learinng/Rag_A_B_Testing/ehr'
CACHE_FILE = 'ehr_cache.parquet'

print(f"Loading files from '{EHR_DATA_FOLDER}'...")
all_files = glob.glob(os.path.join(EHR_DATA_FOLDER, '*.txt'))
patient_data = []

for file_path in all_files:
    patient_id = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    patient_data.append({'patient_id': patient_id, 'content': content})

print(f"Loaded {len(patient_data)} files. Converting to DataFrame...")
df = pd.DataFrame(patient_data)

print(f"Saving to optimized cache file: '{CACHE_FILE}'...")
df.to_parquet(CACHE_FILE, compression='gzip')

print("Pre-processing complete!")
