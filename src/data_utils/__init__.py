import pandas as pd

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

def preprocess_data(threshold, depth, exclude = [], verbose: bool = True):
    global SECTION_KEYS
    global DATASET
    data = DATASET.data
    ref = DATASET.ref
    keys = SECTION_KEYS
    for key in exclude:
       print(key)
       del keys[key]
    return partition_by_pattern(ref, keys)
