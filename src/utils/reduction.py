import pandas as pd
import numpy as np
import re


def passes_sparsity(column, percent=0.95):
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


def reduce_data(data, percent=0.95):
    """Given a dataset return the columns which pass the given sparsity threshold.

    Args:
        data (dataframe): Data to be reduced.
        percent (float): Percentile threshold for the sparcity of the data.

    Returns:
        dataframe: The reduced data.

    """
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
    return matches_regex(">>|;|\|", string)


def has_concentrations(string):
    """Returns True if the given string has concentrations in it.

    Used to check which features have concentrations encoded in them.
    """
    return matches_regex("concentrations", string)


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
    pattern_mask = ref['Pattern'].apply(is_pattern)
    concentrations_mask = ref['Field'].apply(has_concentrations)
    valid_patterns = pattern_mask & ~concentrations_mask
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
