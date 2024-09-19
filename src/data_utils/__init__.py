import os
import pandas as pd
import numpy as np
import copy

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

class PerovskiteData():
   ref = None
   data = None
   ref_file = 'pdp_units_data.xlsx'
   database_file = 'Perovskite_database.csv'
   def load_data(self):
      if self.data is None:
         print("Loading Perovskite Data...")
         database_path = os.path.join('data/', self.database_file)
         if not os.path.exists(database_path):
            database_path = os.path.join('../', database_path)
         if not os.path.exists(database_path):
            database_path = os.path.join('../', database_path)
         print(database_path)
         self.data = pd.read_csv(database_path, low_memory=False)
         self.data.replace(NAN_EQUIVALENTS, inplace=True)
      
      if self.ref is None:
         print("Loading Reference Data...")
         ref_path = os.path.join('data/', self.ref_file)
         if not os.path.exists(ref_path):
            ref_path = os.path.join('../', ref_path)
         if not os.path.exists(ref_path):
            ref_path = os.path.join('../', ref_path)
         print(ref_path)
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

def preprocess_data(threshold, depth, exclude_sections = [], exclude_cols = [], verbose: bool = True):
   global SECTION_KEYS
   global DATASET

   print(os.getcwd())
   DATASET.load_data()
   print("Preprocessing Data...")
   print(f"Threshold: {threshold}, Depth: {depth}")
   data = DATASET.data
   ref = DATASET.ref

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
   return data, _column_selector(expanded_data, nonpatterned_data)

def display_scores(scores):
   print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))