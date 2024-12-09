## Boilerplate code to import util scripts ##
import os
import sys

# Add current directory to system path
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

# Import util scripts
from utils import *
print("Data utils loaded")