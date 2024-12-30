"""This module contains methods to decode layer data from the strings given by certain features.

Many features in the Perovskite Database contain information about the device layers encoded using patterns:
    Cell_stack_sequence: "SLG | FTO | TiO2-c | TiO2-mp | Perovskite | Spiro-MeOTAD | Au"
    ETL_deposition_synthesis_atmosphere: "Vacuum | Vacuum >> Vacuum"

The decoding methods are used to expand this layer data into individual features. The **Depth Threshold**, one of the preprocessing hyperparameters, is used to restrict the sparcity of these features.

From the Perovskite database reference document:
    ### The vertical bar, i.e. (‘ | ‘)
    If a filed contains data for more than one layer, the data belonging to the different layers is separated by a
    vertical bar with a space on both sides, i.e. (‘ | ‘)
    Layers are sorted left to right with the substrate first, i.e. to the left.
    
    ### The semicolon. i.e. (‘ ; ‘)
    If several materials, solvents, gases, etc. are occurring in one layer or during one reaction step, e.g. A and
    B, are listed in alphabetic order and separated with semicolons, as in (A; B)
    
    ### The double forward angel bracket, i.e. (‘ >> ‘)
    When a layer in a stack is deposited, there may be more than one reaction step involved. If that is the case,
    the information concerning the different reaction steps, e.g. A, and B, are separated by a double forward
    angel bracket with one blank space on both sides, as in (‘A >> B‘)
"""
import re
import pandas as pd
import numpy as np
from bigtree import Node

DEPTH_NAMES = {
    0: 'Layer',
    1: 'Feature',
    2: 'Deposition'
}
DEPTH_DELIM = {
    0: '|',
    1: ';',
    2: '>>'
}


def extract_features(seq, delim=';'):
    if isinstance(seq, str):
        seq = seq.replace(' ', '')
        # Unknown sequences return no feature
        if seq in ['Unknown', 'nan']:
            return []
        # Split the sequence into features
        layers = seq.split(delim)
        return layers
    else:
        return []


def generate_children(data, delim):
    def extract(x): return extract_features(x, delim)
    children = pd.DataFrame(data.apply(extract).to_list())
    return children


def feature_counts(children, count_none=True):
    # Create a bit matrix where child data exists
    # The rows of this matrix encode the layers which this data exists at
    bitmat = children.notna().to_numpy()
    (a, b) = bitmat.shape

    # Adds a column of True values at the beginning of the matrix
    # All data exists in the 0th layer (not in the stack sequence)
    if count_none:
        bitmat = np.insert(bitmat, 0, np.ones((1, a), dtype='bool'), axis=1)
        b += 1

    # Transform the rows to arrays of uint8's so bitwise operations can be used
    packedbitmat = np.packbits(bitmat, axis=1)

    # x & -x extracts the furthest 1 bit to the right setting all other bits to 0
    # This is equivalent to extracting the highest layer a datum exists at
    # The bitwise expression is then expanded back into a bit matrix
    expandedbitmat = np.unpackbits(
        packedbitmat & -packedbitmat, axis=1, count=b)

    # Summing along the columns gives the counts for each layer
    return expandedbitmat.sum(axis=0)


def percent_index(arr, percent=0.95):
    # Computes the index where a given percent of data falls within (Defaults to 95%)
    total = arr.sum()
    target = total*percent
    cumulative = np.cumsum(arr)
    return np.where(cumulative >= target)[0][0]

# Filters children within a given percentile


def filter_children(children, percent):
    counts = feature_counts(children)
    index = percent_index(counts, percent)
    children = children[children.columns[0:index]]
    return children


def feature_tree(name, data, parent=None, iter=0, max_iter=len(DEPTH_NAMES), is_only=False, percent=1.0):
    if is_only:
        root = Node(DEPTH_NAMES[iter-1], data=data, parent=parent)
    else:
        root = Node(name, data=data, parent=parent)
    if iter >= max_iter:
        return

    children = generate_children(data, delim=DEPTH_DELIM[iter])
    n_children = np.shape(children)[1]

    # Pruning
    if (percent < 1.0) and (n_children > 0):
        children = filter_children(children, percent)
        n_children = np.shape(children)[1]

    is_only = n_children <= 1
    for child in children:
        feature_tree(
            f"{DEPTH_NAMES[iter]}_{child}",
            children[child],
            root,
            iter+1,
            max_iter,
            is_only,
            percent
        )
    return root


def _set_name(name, leaf):
    if name is None:
        return leaf.name
    else:
        return f"{leaf.name}_{name}"


def generate_name(leaf, name=None):
    if leaf.parent is None:
        name = _set_name(name, leaf)
        return name

    is_match = re.search(r"_(\d+)", leaf.name)
    if is_match:
        name = _set_name(name, leaf)

    return generate_name(leaf.parent, name)


def leaf_matrix(root):
    leaves = np.array([leaf.data.to_list() for leaf in root.leaves])
    columns = [generate_name(leaf) for leaf in root.leaves]
    return pd.DataFrame(leaves.transpose(), columns=columns)


def expand_data(name: str, data, percent: float = 1.0, verbosity: int = 0):
    tree = feature_tree(name, data, percent=percent)
    if verbosity > 2:
        tree.show()
    return leaf_matrix(tree)


def expand_dataset(data, features=None, percent: float = 1.0, verbosity: int = 0):
    if features is None:
        features = data.columns
    expanded_data = []
    if type(features) is dict:
        for (feature, name) in features.items():
            expanded_data.append(expand_data(
                name, data[feature], percent=percent, verbosity=verbosity))
    else:
        for feature in features:
            expanded_data.append(expand_data(
                feature, data[feature], percent=percent, verbosity=verbosity))
    return pd.concat(expanded_data, axis=1)
