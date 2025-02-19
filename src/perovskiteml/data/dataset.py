import os
import json
import pyarrow as pa
import pyarrow.parquet as pq

from .database import DATABASE
from ..utils import EXPAND_DIR
from ..preprocessing.reduction import prune_by_sparsity, collect_features, remove_features, section_features
from ..preprocessing.expansion import expand_sort
from ..preprocessing.preprocess import _to_numeric


def has_dataset(target, in_path=EXPAND_DIR):
    """Checks if a dataset exists for a given target feature.
    
    Args:
        target (str): Name of the target feature.
        in_path (str, optional): Path to the directory containing the dataset.
            Defaults to EXPAND_DIR.

    Returns:
        bool: True if the dataset exists. False otherwise.
    
    """
    target_path = os.path.join(in_path, target)
    if not os.path.exists(target_path):
        return False
    dataset_path = os.path.join(target_path, "dataset.parquet")
    if not os.path.exists(dataset_path):
        return False
    features_path = os.path.join(target_path, "features.json")
    if not os.path.exists(features_path):
        return False

    return True


def save_dataset(data, features: dict, target: str, out_path=EXPAND_DIR):
    """Saves the data and metadata for a given target feature.

    See the `README.md` in `./data` for more information about the file structure.
    
    Args:
        data (dataframe): The data.
        features (dict): The features.
        target (str): Name of the target feature.
        out_path (str, optional): Path to the directory to save the dataset.
            Defaults to EXPAND_DIR.
            
    Returns:
        None
    
    """
    target_path = os.path.join(out_path, target)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    dataset_path = os.path.join(target_path, "dataset.parquet")
    table = pa.Table.from_pandas(data)
    metadata = table.schema.metadata or {}
    metadata.update({"target": target})
    pq.write_table(table, dataset_path)

    features_path = os.path.join(target_path, "features.json")
    with open(features_path, "w") as file:
        json.dump(features, file, indent=4)


def load_dataset(target: str, in_path=EXPAND_DIR):
    """Loads the data and metadata for a given target feature.
    
    Args:
        target (str): Name of the target feature.
        in_path (str, optional): Path to the directory to load the dataset.
            Defaults to EXPAND_DIR.

    Returns:
        dataframe: The data.
        dict: The features.
    
    """
    target_path = os.path.join(in_path, target)
    dataset_path = os.path.join(target_path, "dataset.parquet")
    features_path = os.path.join(target_path, "features.json")
    table = pq.read_table(dataset_path)
    data = table.to_pandas()
    with open(features_path, "r") as file:
        features = json.load(file)
    return data, features


def has_reference(path=EXPAND_DIR):
    """Checks if the database reference exists.
    
    Args:
        path (str, optional): Path to the directory containing the reference.
            Defaults to EXPAND_DIR.

    Returns:
        bool: True if the reference exists. False otherwise.
    
    """
    reference_path = os.path.join(path, "reference.json")
    if not os.path.exists(reference_path):
        return False
    return True


def save_reference(ref, out_path=EXPAND_DIR):
    """Saves a database reference.
    
    See the `README.md` in `./data` for more information about the file structure.
    
    Args:
        ref (dict): The reference.
        out_path (str, optional): Path to the directory to save the reference.
            Defaults to EXPAND_DIR.

    Returns:
        None
    
    """
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    reference_path = os.path.join(out_path, "reference.json")
    with open(reference_path, "w") as file:
        json.dump(ref, file, indent=4)


def load_reference(in_path=EXPAND_DIR):
    """Loads the database reference.
    
    Args:
        in_path (str, optional): Path to the directory to load the reference.
            Defaults to EXPAND_DIR.

    Returns:
        dict: The reference.
    
    """
    reference_path = os.path.join(in_path, "reference.json")
    with open(reference_path, "r") as file:
        ref = json.load(file)
    return ref


def generate_reference(refs):
    """Generates a database reference.
    
    Args:
        refs (dataframe): An expanded representation of the database reference.

    Returns:
        dict: The reference.
    
    """
    return {
        ref: refs[ref]['Field'].to_list()
        for ref in refs if ref != 'Full Table'
    }


def generate_dataset(target, database=DATABASE, save=True):
    """Generates an expanded dataset for a given target feature.
    
    Args:
        target (str): Name of the target feature.
        database (PerovskiteDatabase, optional): An instance of the Perovskite Dataset.
            Defaults to DATABASE.
        save (bool, optional): Whether to save the dataset.
            Defaults to True.

    Returns:
        dataframe: The data.
        dict: The data features.
        dict: The database reference.
    
    """
    database.load_data()
    x, _ = database.get_Xy(database.data, target)
    if has_reference():
        ref = load_reference()
    else:
        ref = generate_reference(database.ref)
    data, features = expand_sort(x, database.ref['Full Table'])
    data = data.apply(_to_numeric)
    if save:
        save_reference(ref)
        save_dataset(data, features, target)
    return data, features, ref

def generate_groups(target, group, database=DATABASE):
    """Generates a list of groups for a given target feature and group feature.
    
    Args:
        target (str): Name of the target feature.
        group (str): Name of the feature to group by.
        
    Return:
        list: The list of groups.
    
    """
    database.load_data()
    x, _ = database.get_Xy(database.data, target)
    return x[group].astype(str).to_list()
    

class DataSet():
    """Stores the expanded perovskite data and metadata for a given target feature.
    
    Attributes:
        target (str): Name of the target feature.
        data (dataframe): The expanded data.
        reference (dict): The database reference.
        all_features (dict): The full set of features for the expanded data.
        features (dict): A reduced set of features generated during preprocessing.
        groups (list): A list of groups for the target feature.
    
    """
    def __init__(self, target, group_by=None):
        self.target = target
        self.data = None
        self.reference = None
        self.all_features = None
        self.features = None
        if group_by is not None:
            self.groups = generate_groups(target, group_by)
        else:
            self.groups = group_by

    def reset_features(self):
        """Resets the reduced set of features to the full set of features.
        
        Returns:
            None
        
        """
        self.features = self.all_features
        return

    def collect_features(self):
        """Collects the features from the reduced set of features.
        
        Returns:
            list: The list of features.
        
        """
        return collect_features(self.features)

    def get_dataset(self, database=DATABASE, save: bool = True):
        """Loads the data and metadata for a given target feature.
        
        Stores the data and metadata in the class attributes.

        Args:
            database (PerovskiteDatabase, optional): An instance of the Perovskite Dataset.
                Defaults to DATABASE.
            save (bool, optional): Whether to save the dataset.
                Defaults to True.

        Returns:
            None
        
        """
        if has_dataset(self.target):
            self.data, self.all_features, = load_dataset(self.target)
            self.reference = load_reference()
            self.reset_features()
            return

        self.data, self.all_features, self.reference = generate_dataset(
            self.target, database=database, save=save)
        self.reset_features()
        return

    def remove_features(self, features):
        """Removes features from the reduced set of features.
        
        Args:
            features (list): The list of features to remove.

        Returns:
            None
        
        """
        self.features = remove_features(self.features, features)
        return

    def remove_sections(self, sections):
        """Removes an entire section of features from the reduced set of features.
        
        Args:
            sections (list): The list of sections to remove.

        Returns:
            None
        
        """
        self.features = remove_features(
            self.features,
            section_features(sections, self.reference)
        )
        return

    def remove(self, sections=[], features=[]):
        """Removes both sections and features from the reduced set of features.

        Args:
            sections (list): The list of sections to remove.
                Defaults to [].
            features (list): The list of features to remove.
                Defaults to [].

        Returns:
            None

        """
        remove_list = section_features(sections, self.reference)
        remove_list.extend(features)
        self.remove_features(remove_list)
        return

    def prune_by_sparsity(self, threshold):
        """Prunes the reduced set of features by sparsity.
        
        Args:
            threshold (float): The sparsity threshold.

        Returns:
            None
        
        """
        self.features = prune_by_sparsity(self.features, threshold)
        return

    def get_Xy(self):
        """Returns the reduced data and the target series.

        Returns:
            dataframe: The reduced data.
            series: The target series.
        
        """
        feature_list = self.collect_features()
        return self.data[feature_list], self.data[self.target]

    def preprocess(self, threshold=None, exclude_sections=[], exclude_cols=[]):
        """Preprocesses the data.
        
        If an unseen target is used to generate the preprocessed dataset, it is saved for future use. Otherwise, the previously generated file is loaded and returned instead.
        
        Args:
            threshold (float, optional): The sparsity threshold.
                Defaults to None.
            exclude_sections (list, optional): The list of sections to exclude.
                Defaults to [].
            exclude_cols (list, optional): The list of columns to exclude.
                Defaults to [].

        Returns:
            dataframe: The preprocessed data.
            series: The target series.

        """
        self.get_dataset()
        ##-- TODO: Add selective preprocessing step here --##
        self.remove(sections=exclude_sections, features=exclude_cols)
        if threshold is not None:
            self.prune_by_sparsity(threshold)
        return self.get_Xy()