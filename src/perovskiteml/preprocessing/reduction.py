import pandas as pd
import numpy as np
import bisect
import re
from collections import deque
from bigtree import dict_to_tree, tree_to_dict


def find_sparsity(column):
    """Finds the sparsity of a column of data.

    Args:
        column (series): Column to be checked.

    Returns:
        float: The sparsity of the data.
    
    """
    total = len(column)
    return float(column.isna().sum()/total)


def sort_by_sparsity(features):
    """Sorts a dictionary of features by their sparsity.
    
    Args:
        features (dict): Dictionary of features to be sorted.
        
    Returns:
        dict: Sorted dictionary of features.

    """
    sparsity = [feature['sparsity'] for feature in features]
    index = np.argsort(sparsity)
    features = [features[i] for i in index]
    return features


def prune_by_sparsity(features, threshold):
    """Prunes a dictionary of features by their sparsity.
    
    Args:
        features (dict): Dictionary of features to be pruned.
        threshold (float): Threshold for the sparsity of the data.
        
    Returns:
        dict: Pruned dictionary of features.
        
    """
    index = bisect.bisect_left(
        features, threshold, key=lambda x: x['sparsity'])
    return features[:index]


def passes_sparsity(column, percent=0.0):
    """Checks if a column of data passes the sparsity threshold.

    Args:
        column (series): Column to be checked.
        percent (float): Percentile threshold for the sparcity of the data.
            Defaults to 0.95.

    Returns:
        bool: True if the data sparcity passes the threshold. False otherwise.

    """
    total = len(column)
    target = total*percent
    return column.notna().sum() >= target


def reduce_data(data, percent=0.0):
    """Given a dataset return the columns which pass the given sparsity threshold.

    Args:
        data (dataframe): Data to be reduced.
        percent (float): Percentile threshold for the sparcity of the data.

    Returns:
        dataframe: The reduced data.

    """
    if percent <= 0.0:
        return data
    valid_columns = data.apply(passes_sparsity, args=(percent,))
    valid_mask = valid_columns.keys()[valid_columns.values]
    return data[valid_mask]


def matches_regex(rgx, string):
    """Returns True if the given string matches the given regex string."""
    if pd.isna(string):
        return False
    result = re.search(rgx, string)
    return bool(result)


def is_pattern(string):
    """Returns True if the given string is a pattern.

    Used to check which features have layer data encoded in them.

    Examples of valid patterns:
        "[Mat.1; Mat.2; ... | Mat.3; ... | Mat.4 | ...]"
        "[Gas1; Gas2 >> Gas3; ... >> ... | Gas4 >> … | Gas5 | ... ]"
    """
    return matches_regex(r">>|;|\|", string)


def has_concentrations(string):
    """Returns True if the given string has concentrations in it.

    Used to check which features have concentrations encoded in them.
    """
    return matches_regex("concentrations", string)


def is_valid_pattern(ref):
    """Returns a mask of the features which encode layer data with patterns.
    
    Args:
        ref (dataframe): The reference data for features.

    Returns:
        series: A mask of the features which encode layer data with patterns.

    """
    pattern_mask = ref['Pattern'].apply(is_pattern)
    concentrations_mask = ref['Field'].apply(has_concentrations)
    return pattern_mask & ~concentrations_mask


def get_valid_patterns(ref, return_invalid: bool = True):
    """Finds the features which encode layer data with patterns. Returns the feature names.

    Args:
        ref (dataframe): The reference data for features.
        return_invalid (bool): Option to return a second set containing the names of the features which don't contain patterns.
            Default True.

    Returns:
        list: patterned feature names.
        list: nonpatterned features names if return_invalid is True.
    """
    valid_patterns = is_valid_pattern(ref)
    if return_invalid:
        return ref[valid_patterns]['Field'], ref[~valid_patterns]['Field']
    else:
        return ref[valid_patterns]['Field']


def partition_by_pattern(refs, keys):
    """Given a subset of features, partition the features into a patterned and non-patterned set.

    Args:
        refs (dict of dataframe): A set of references for each section of features.
            Sections include "Hole transport layer", "The perovskite", etc.
        keys (list): List of section names to be included in the partitioning.

    Return:
        list: patterned features.
        list: nonpatterned features.
    """
    patterned = []
    nonpatterned = []
    for key in keys:
        ref = refs[key]
        pat, nonpat = get_valid_patterns(ref)
        patterned.append(pat)
        nonpatterned.append(nonpat)
    patterned = pd.concat(patterned)
    nonpatterned = pd.concat(nonpatterned)
    return patterned, nonpatterned


def collect_features(features):
    """Collects the expanded features from a dictionary of features.

    Args:
        features (dict): Dictionary of features.

    Returns:
        list: List of expanded features.

    """
    feature_array = []
    for feature_set in features:
        feature_array.extend(feature_set['children'])
    return feature_array


def section_features(sections, ref):
    """Gets the features for a given section from a reference dictionary.
    
    Args:
        sections (list): List of sections to be included.
        ref (dict): The reference data for features.

    Returns:
        list: List of features for the given section.

    """
    return [feature for section in sections for feature in ref[section]]


def remove_features(features, remove):
    """Removes features from a dictionary of features.

    Args:
        features (dict): Dictionary of features.
        remove (list): List of features to be removed.

    Returns:
        dict: Dictionary of features without the removed features.
        
    """
    return [feature for feature in features if feature['parent'] not in remove]


def percent_index(arr, percent=0.95):
    """Finds the index where a given percent of data falls within.
    
    Args:
        arr (array): The bitarray which describes the sparsity of the data.
        percent (float, optional): The percentile threshold.
            Defaults to 0.95.

    Returns:
        int: The index where a given percent of data falls within.

    """
    total = arr.sum()
    target = total*percent
    cumulative = np.cumsum(arr)
    return np.where(cumulative >= target)[0][0]

def remove_nodes_bfs(root, n):
    if not root:
        return None

    # Use a queue for BFS traversal
    queue = deque([root])

    while queue:
        node = queue.popleft()
        
        # Remove children that do not satisfy the condition
        # Only children are preserved
        children = node.children
        if len(children) > 1:
            counts = np.array([child.counts for child in children])
            index = percent_index(counts, n)
            if index < 1:
                index += 1
            
            node.children = children[0:index]
        
        for child in node.children:
            queue.append(child)

    return root

def prune_and_combine(feature_list, coverage_threshold=0.95):
    for feature in feature_list:
        tree_dict = feature["tree"]
        if not tree_dict:
            continue
        tree = dict_to_tree(tree_dict)
        pruned_tree = remove_nodes_bfs(tree, coverage_threshold)
        pruned_features = [leaf.name for leaf in pruned_tree.leaves]
        feature["children"] = pruned_features
        feature["tree"] = tree_to_dict(pruned_tree)
    return feature_list