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
    def load_data(self):
      if self.data is None:
        print("Loading Perovskite Data...")
        self.data = pd.read_csv('../data/Perovskite_database_content_all_data_040524.csv', low_memory=False)
        self.data.replace(NAN_EQUIVALENTS, inplace=True)
      
      if self.ref is None:
        print("Loading Reference Data...")
        self.ref = pd.read_excel('../data/pdp_units_data.xlsx', sheet_name=None)
         
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
    return pd.concat([expanded_data, nonpatterned_data], axis=1), _column_selector(expanded_data, nonpatterned_data)

