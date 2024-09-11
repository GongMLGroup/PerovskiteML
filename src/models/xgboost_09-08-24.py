import pandas as pd
import numpy as np

# import custom scripts
from data_utils import *

# import data
DATASET.load_data()
print(partition_by_pattern(DATASET.ref, SECTION_KEYS))