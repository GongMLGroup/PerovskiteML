import pandas as pd
import numpy as np
from bigtree import Node

def extract_features(seq, delim=';'):
  if isinstance(seq, str):
    seq = seq.replace(' ', '')
    # Unknown sequences return no layers
    if seq == 'Unknown':
      return []
    # Split the sequence into layers
    layers = seq.split(delim)
    return layers
  else:
    return []

def extract_layers(seq):
  return extract_features(seq, delim='|')

def feature_tree(name, data, feature_name="Feature"):
  # Depth 0: Full Dataset
  # Depth 1: Layers
  # Depth 2: SubLayers
  # Depth 3: Features (Leaves)
  root = Node(name, data=data)
  for col in data:
    parent = Node(col, data=data[col], parent=root)
    # Extract sub-layers and set missing values to None
    df = pd.DataFrame(data[col].apply(extract_layers).to_list())
    for sublayer in df:
      child = Node(f"Layer_{sublayer}", data=df[sublayer], parent=parent)
      # Extract features and set missing values to None
      df2 = pd.DataFrame(df[sublayer].apply(extract_features).to_list())
      for feat in df2:
        Node(f"{feature_name}_{feat}", data=df2[feat], parent=child)
  return root

def layer_counts(root):
  # Create a bit matrix where child data exists
  # The columns of this matrix encode the layers which this data exists at
  bitmat = np.array([child.data.notna().to_list() for child in root.children])
  (a, b) = bitmat.shape

  # Adds a row of True values at the beginning of the matrix
  # All data exists in the 0th layer (not in the stack sequence)
  bitmat = np.insert(bitmat, 0, np.ones((1,b), dtype='bool'), axis=0)

  # Transform the columns to arrays of uint8's so bitwise operations can be used
  packedbitmat = np.packbits(bitmat, axis=0)

  # x & -x extracts the furthest 1 bit to the right setting all other bits to 0
  # This is equivalent to extracting the highest layer a datum exists at
  # The bitwise expression is then expanded back into a bit matrix
  expandedbitmat = np.unpackbits(packedbitmat & -packedbitmat, axis=0, count=a+1)

  # Summing along the rows gives the counts for each layer
  return range(a+1), expandedbitmat.sum(axis=1)

def percent_index(arr, percent=0.95):
  # Computes the index where a given percent of data falls within (Defaults to 95%)
  total = arr.sum()
  target = total*percent
  cumulative = np.cumsum(arr)
  return np.where(cumulative >= target)[0][0]

def prune_tree(root, percent=0.95):
  # This function mutates the input tree and prunes it based on the the amount
  # of data which falls within a given percentile (Defaults to 95%)
  if len(root.children) == 0:
    return
  _, counts = layer_counts(root)
  index = percent_index(counts, percent)
  root.children = root.children[0:index]
  for child in root.children:
    prune_tree(child, percent)

def generate_feature_matrix(tree):
  features = np.array([leaf.data.to_list() for leaf in tree.leaves])
  columns = [f"{leaf.parent.parent.name}_{leaf.parent.name}_{leaf.name}" for leaf in tree.leaves]
  return pd.DataFrame(features.transpose(), columns=columns)
