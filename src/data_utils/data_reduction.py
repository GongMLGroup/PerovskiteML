import pandas as pd
import numpy as np
import re

def passes_sparsity(column, percent=0.95):
    total = len(column)
    target = total*percent
    return column.notna().sum() >= target

def reduce_data(data, percent=0.95):
  valid_columns = data.apply(passes_sparsity, args=(percent,))
  valid_mask = valid_columns.keys()[valid_columns.values]
  return data[valid_mask]

def matches_regex(rgx, string):
  if pd.isna(string):
    return False
  result = re.search(rgx, string)
  return bool(result)

def is_pattern(string):
  return matches_regex(">>|;|\|", string)

def has_concentrations(string):
  return matches_regex("concentrations", string)

def get_valid_patterns(ref, return_invalid: bool=True):
  pattern_mask = ref['Pattern'].apply(is_pattern)
  concentrations_mask = ref['Field'].apply(has_concentrations)
  valid_patterns = pattern_mask & ~concentrations_mask
  if return_invalid:
    return ref[valid_patterns]['Field'], ref[~valid_patterns]['Field']
  else:
    return ref[valid_patterns]['Field']
  
def partition_by_pattern(refs, keys):
  patterned = []
  nonpatterned = []
  for key in keys:
    ref = refs[key]
    pat, nonpat = get_valid_patterns(ref)
    # print(pat, nonpat)
    patterned.append(pat)
    nonpatterned.append(nonpat)
  patterned = pd.concat(patterned)
  nonpatterned = pd.concat(nonpatterned)
  return patterned, nonpatterned