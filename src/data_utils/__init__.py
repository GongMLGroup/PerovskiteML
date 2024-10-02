import os
import copy
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

## This File acts a pseudo package, allowing useful scripts to be easily imported
from .data_expansion import *
from .data_reduction import *

SECTION_KEYS = {
    "Reference information": "Ref",
    "Cell definition": "Cell",
    "Module definition": "Module",
    "Substrate": "Substrate",
    "Electron transport layer": "ETL",
    "The perovskite": "Perovskite",
    "Perovskite deposition": "Perovskite_Dep",
    "Hole transport layer": "HTL",
    "Back contact": "Backcontact",
    "Additional layers": "Add_lay",
    "Encapsulation": "Encapsulation",
    "JV data": "JV",
    "Stabilised efficiency": "Stabilised",
    "Quantum efficiency": "EQE",
    "Stability": "Stability",
    "Outdoor testing": "Outdoor",
  }

NAN_EQUIVALENTS = {
    'Unknown': None,
    'unknown': None,
    'None': None,
    'none': None,
    'NULL': None,
    'Null': None,
    'null': None,
    'NAN': None,
    'NaN': None,
    'Nan': None,
    'nan': None,
    'NAN; NAN': None,
    'NaN; NaN': None,
    'Nan; Nan': None,
    'nan; nan': None
  }

def _search_parents(path, depth=2):
   for x in range(depth):
      if os.path.exists(path):
         return path
      path = os.path.join('../', path)
   return path

class PerovskiteData():
   ref = None
   data = None
   ref_file = 'pdp_units_data.xlsx'
   database_file = 'Perovskite_database.csv'
   def load_data(self):
      if self.data is None:
         print("Loading Perovskite Data...")
         database_path = os.path.join('data/', self.database_file)
         database_path = _search_parents(database_path)
         self.data = pd.read_csv(database_path, low_memory=False)
         self.data.replace(NAN_EQUIVALENTS, inplace=True)
      
      if self.ref is None:
         print("Loading Reference Data...")
         ref_path = os.path.join('data/', self.ref_file)
         ref_path = _search_parents(ref_path)
         self.ref = pd.read_excel(ref_path, sheet_name=None)
         
      print("Data Initialized.")
DATASET = PerovskiteData()

def _column_selector(pat, nonpat):
   patterned = [col for col in pat]
   categorical = []
   numerical = []
   for col in nonpat:
      col_types = nonpat[col].apply(type)
      if np.any((col_types == bool) | (col_types == object) | (col_types == str)):
         categorical.append(col)
      else:
         numerical.append(col)
   return {
      'patterned': patterned,
      'categorical': categorical,
      'numerical': numerical
   }

def _to_numeric(col):
    col_types = col.apply(type)
    if not np.any((col_types == bool)):
        return pd.to_numeric(col, errors='ignore')
    return col

def hash_params(params: dict):
   param_str = '_'.join(f'{key}={value}' for key, value in sorted(params.items()))
   return hashlib.md5(param_str.encode()).hexdigest()

def preprocess_data(threshold, depth, exclude_sections = [], exclude_cols = [], verbose: bool = True):
   global SECTION_KEYS
   global DATASET

   params = {
      'threshold': threshold,
      'depth': depth,
      'xsections': exclude_sections,
      'xcols': exclude_cols,
   }

   # Generate file name from parameters
   file_hash = hash_params(params)
   file_name = f'expanded_data_d{depth}_t{threshold}_{file_hash}.parquet'
   file_path = _search_parents('data/preprocessed')
   file_path = os.path.join(file_path, file_name)

   # Check if file exists
   if os.path.exists(file_path):
      print(f'File already exists: {file_path}')
      print('Loading Data...')
      table = pq.read_table(file_path)
      return table.to_pandas(), None

   print(f"File does not exist, preprocessing and saving to {file_path}")
   DATASET.load_data()
   print("Preprocessing Data...")
   print(f"Threshold: {threshold}, Depth: {depth}")
   data = DATASET.data
   ref = DATASET.ref

   # Remove excluded keys
   keys = copy.copy(SECTION_KEYS)
   for key in exclude_sections:
      del keys[key]

   patterned, nonpatterned = partition_by_pattern(ref, keys)

   patterned_data = reduce_data(data[patterned], percent=threshold)
   patterned_data.drop(columns=exclude_cols, inplace=True, errors='ignore')

   nonpatterned_data = reduce_data(data[nonpatterned], percent=threshold)
   nonpatterned_data.drop(columns=exclude_cols, inplace=True, errors='ignore')

   expanded_data = expand_dataset(patterned_data, percent=depth, verbose=verbose)
   print("Data Preprocessed.")
   data = pd.concat([expanded_data, nonpatterned_data], axis=1)
   data.replace(NAN_EQUIVALENTS, inplace=True)
   data = data.apply(_to_numeric)

   # Save Data
   print(f"Saving Data to {file_path}...")
   table = pa.Table.from_pandas(data)
   metadata = table.schema.metadata or {}
   metadata.update({key.encode(): str(value).encode() for key, value in params.items()})
   pq.write_table(table.replace_schema_metadata(metadata), file_path)
   print("Data Saved.")

   return data, _column_selector(expanded_data, nonpatterned_data)

def display_scores(scores):
   print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))