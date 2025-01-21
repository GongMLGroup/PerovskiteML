import os
import json
import pyarrow as pa
import pyarrow.parquet as pq

from .database import DATASET
from .fileutils import EXPAND_DIR
from .reduction import prune_by_sparsity, collect_features, remove_features, section_features
from .expansion import expand_sort

def has_dataset(target, in_path=EXPAND_DIR):
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
    target_path = os.path.join(in_path, target)
    dataset_path = os.path.join(target_path, "dataset.parquet")
    features_path = os.path.join(target_path, "features.json")
    table = pq.read_table(dataset_path)
    data = table.to_pandas()
    with open(features_path, "r") as file:
        features = json.load(file)
    return data, features


def has_reference(path=EXPAND_DIR):
    reference_path = os.path.join(path, "reference.json")
    if not os.path.exists(reference_path):
        return False
    return True


def save_reference(ref, out_path=EXPAND_DIR):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    reference_path = os.path.join(out_path, "reference.json")
    with open(reference_path, "w") as file:
        json.dump(ref, file, indent=4)


def load_reference(in_path=EXPAND_DIR):
    reference_path = os.path.join(in_path, "reference.json")
    with open(reference_path, "r") as file:
        ref = json.load(file)
    return ref


def generate_reference(refs):
    return {
        ref: refs[ref]['Field'].to_list()
        for ref in refs if ref != 'Full Table'
    }


def generate_dataset(target, database=DATASET, save=True):
    database.load_data()
    x, _ = database.get_Xy(database.data, target)
    if has_reference():
        ref = load_reference()
    else:
        ref = generate_reference(database.ref)
    data, features = expand_sort(x, database.ref['Full Table'])
    if save:
        save_reference(ref)
        save_dataset(data, features, target)
    return data, features, ref

class DataSet():
    def __init__(self, target):
        self.target = target
        self.data = None
        self.reference = None
        self.all_features = None
        self.features = None

    def reset_features(self):
        self.features = self.all_features
        return

    def collect_features(self):
        return collect_features(self.features)

    def get_dataset(self, database=DATASET, save: bool = True):
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
        self.features = remove_features(self.features, features)
        return

    def remove_sections(self, sections):
        self.features = remove_features(
            self.features,
            section_features(sections, self.reference)
        )
        return

    def remove(self, sections=[], features=[]):
        remove_list = section_features(sections, self.reference)
        remove_list.extend(features)
        self.remove_features(remove_list)
        return

    def prune_by_sparsity(self, threshold):
        self.features = prune_by_sparsity(self.all_features, threshold)
        return

    def get_Xy(self):
        feature_list = self.collect_features()
        return self.data[feature_list], self.data[self.target]

    def preprocess(self, threshold=None, exclude_sections=[], exclude_cols=[]):
        self.get_dataset()
        ##-- TODO: Add selective preprocessing step here --##
        self.remove(sections=exclude_sections, features=exclude_cols)
        if threshold is not None:
            self.prune_by_sparsity(threshold)
        return self.get_Xy()