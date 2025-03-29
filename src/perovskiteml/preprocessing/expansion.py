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
from bigtree import Node, tree_to_dict
from .reduction import find_sparsity, sort_by_sparsity, is_valid_pattern


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
    """Converts a sequence of encoded data into a list of features.

    Args:
        seq (str): The sequence of encoded data.
        delim (str): The delimiter used to separate features.
            Defaults to ';'. (The Feature Layer Delimiter)

    Returns:
        list: The list of features.

    """
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
    """Generates the child features for a column of data.

    Each column of the generated dataframe is a child feature.

    Args:
        data (series): The column of data.
        delim (str): The delimiter used to separate features.

    Returns:
        dataframe: The child features.

    """
    def extract(x): return extract_features(x, delim)
    children = pd.DataFrame(data.apply(extract).to_list())
    return children


def feature_counts(children, count_none=True):
    """Finds the maximum layer number for each row of data and sums along the rows.

    This counts how many times a maximum layer occurs.

    Args:
        children (dataframe): The child features.
        count_none (bool, optional): Counts the 0th layer as a layer.
            Defaults to True.

    Returns:
        array: The counts for each layer.


    """
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


def _feature_tree(name, data, parent=None, iter=0, max_iter=len(DEPTH_NAMES), is_only=False, counts=0):
    """Generates a tree of features from a column of encoded data.

    Args:
        name (str): The name of the feature.
        data (series): The column of data.
        parent (Node, optional): The parent node.
            Defaults to None.
        iter (int, optional): The current iteration.
            Defaults to 0.
        max_iter (int, optional): The maximum number of iterations.
            Defaults to len(DEPTH_NAMES).
        is_only (bool, optional): If the current node is an only child.
            Defaults to False.
        counts (int, optional): The device counts for each layer.
            Defaults to 0.

    Returns:
        Node: The root node of the tree.

    """
    if is_only:
        root = Node(DEPTH_NAMES[iter-1], data=data,
                    parent=parent, counts=counts)
    else:
        root = Node(name, data=data, parent=parent, counts=counts)
    if iter >= max_iter:
        return

    children = generate_children(data, delim=DEPTH_DELIM[iter])
    n_children = np.shape(children)[1]

    if n_children > 0:
        counts = feature_counts(children)

    is_only = n_children <= 1
    for (i, child) in enumerate(children):
        _feature_tree(
            f"{DEPTH_NAMES[iter]}_{child}",
            children[child],
            root,
            iter+1,
            max_iter,
            is_only,
            int(counts[i])
        )
    return root


def _set_name(name, leaf):
    """Sets the name of a leaf node.

    Args:
        name (str): The name to set.
        leaf (Node): The leaf node.

    Returns:
        str: The name of the leaf node.

    """
    if name is None:
        return leaf.name
    else:
        return f"{leaf.name}_{name}"


def generate_name(leaf, name=None):
    """Recursively generates a name for a leaf node.

    Steps through the nodes to name the feature at the leaf node based on its depth. Examples of names at different depths are:
        - Depth 0: OriginalFeature_Layer_0
        - Depth 1: OriginalFeature_Layer_1_Feature_0
        - Depth 2: OriginalFeature_Layer_1_Feature_2_Deposition_1

    Args:
        leaf (Node): The leaf node.
        name (str, optional): The name to set.
            Defaults to None.

    Returns:
        str: The name of the leaf node.

    """
    if leaf.parent is None:
        name = _set_name(name, leaf)
        return name

    is_match = re.search(r"_(\d+)", leaf.name)
    if is_match:
        name = _set_name(name, leaf)

    return generate_name(leaf.parent, name)


def leaf_matrix(root):
    """Generates a dataframe of the data from the leaf nodes of a tree.

    The columns are labeled with the node names.

    Args:
        root (Node): The root node of the tree.

    Returns:
        dataframe: The dataframe of the data.

    """
    leaves = np.array([leaf.data.to_list() for leaf in root.leaves])
    columns = [generate_name(leaf) for leaf in root.leaves]
    return pd.DataFrame(leaves.transpose(), columns=columns)


def expand_data(name: str, data, verbosity: int = 0):
    """Expands the encoded data from a column of data.

    Args:
        name (str): Name of the feature.
        data (series): The column of data.
        verbosity (int, optional): The verbosity level.
            Defaults to 0.

    Returns:
        dataframe: The expanded data.

    """
    tree = _feature_tree(name, data)
    if verbosity > 2:
        tree.show()
    return tree


def expand_dataset(data, features=None, verbosity: int = 0):
    """Expands the encoded data from a dataset.

    Iterates over each column to expand the data.

    Args:
        data (dataframe): The dataset.
        features (list, optional): The list of features to expand.
            Defaults to None.
        verbosity (int, optional): The verbosity level.
            Defaults to 0.

    Returns:
        dataframe: The expanded dataset.

    """
    if features is None:
        features = data.columns
    expanded_data = []
    if type(features) is dict:
        for (feature, name) in features.items():
            expanded_tree = expand_data(
                name, data[feature], verbosity=verbosity)
            expanded_data.append(leaf_matrix(expanded_tree))
    else:
        for feature in features:
            expanded_tree = expand_data(
                feature, data[feature], verbosity=verbosity)
            expanded_data.append(leaf_matrix(expanded_tree))
    return pd.concat(expanded_data, axis=1)


def expand_sort(data, ref):
    """Expands a dataset and sorts a reference dictionary of the features by sparsity.

    Detects and expands valid encoded features and passes the rest.

    Args:
        data (dataframe): The dataset.
        ref (dataframe): The reference dictionary for the database.

    Returns:
        dataframe: The expanded dataset.
        dict: The sorted reference dictionary of features.

    """
    data.reset_index(drop=True, inplace=True)
    expanded_data = []
    features = []
    valid_pattern_mask = is_valid_pattern(ref)
    for (i, feature) in enumerate(ref['Field']):
        if valid_pattern_mask[i]:
            expanded_tree = expand_data(feature, data[feature])
            expanded_feature = leaf_matrix(expanded_tree)
            expanded_data.append(expanded_feature)
            for leaf in expanded_tree.leaves:
                leaf.name = generate_name(leaf)
            features.append({
                'parent': feature,
                'children': expanded_feature.columns.to_list(),
                'sparsity': find_sparsity(data[feature]),
                'tree': tree_to_dict(expanded_tree, attr_dict={"counts": "counts"})
            })
        else:
            expanded_data.append(data[feature])
            features.append({
                'parent': feature,
                'children': [feature],
                'sparsity': find_sparsity(data[feature]),
                'tree': {}
            })
    expanded_data = pd.concat(expanded_data, axis=1)
    features = sort_by_sparsity(features)
    return expanded_data, features
