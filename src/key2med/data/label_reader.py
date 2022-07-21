import csv
import glob
import logging
import os
import pickle
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import coloredlogs
import numpy as np
import openpyxl
import pandas as pd
import torch

from key2med.data.hierarchy import default_hierarchy, fix_vector
from key2med.utils.logging import tqdm

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Path = str
ImagePath = str
Label = int


class LabelReader:
    data: Dict[ImagePath, Any] = None
    image_paths: List[ImagePath] = None


class UKBLabelReader(LabelReader):
    def __init__(
        self,
        base_path: str,
        extension: str = ".nii",
        max_size: int = None,
        gold_filter: str = None,
        label_filter: List[int] = None,
        cut_filter: bool = None,
    ):

        self.extension = extension
        self.base_path = base_path
        label_file = os.path.join(base_path, "metaInfoAndLabels.csv")
        self.max_size = max_size
        self.label_filter = self.init_label_filter(label_filter, gold_filter)
        self.data, self.index_to_label = self.read_label_csv(label_file, gold_filter, self.label_filter, cut_filter)
        self.label_to_index = {label: index for index, label in enumerate(self.index_to_label)}
        self.label_dim = len(self.index_to_label)
        self.image_paths = list(self.data.keys())

    def init_label_filter(
        self, label_filter: Union[List[int], str], gold_filter: Union[List[int], str]
    ) -> Optional[List[int]]:
        if label_filter is None or label_filter == "full":
            return None
        if label_filter == "gold_all":
            if gold_filter == False:
                print("Warning, gold_filter is set to False, but needs to be True for label_filter gold_all.")
            return list(range(0, 36))
        if label_filter == "gold_small" and gold_filter == "gold":
            return [12, 13, 26, 27, 28, 30]
        if label_filter == "gold_small" and gold_filter == "silver":
            return [36, 37, 38, 39, 40, 41]
        if label_filter == "gold_small" and gold_filter == "gold+silver":
            return [[12, 13, 26, 27, 28, 30], [36, 37, 38, 39, 40, 41]]
        if isinstance(label_filter, list) and isinstance(label_filter[0], int):
            return label_filter
        raise NotImplementedError

    def __getitem__(self, path):
        data = self.data[path]
        return data

    def read_row(self, row: List, label_filter: List[int]) -> Dict[str, Any]:
        """
        Read a single row in the label csv. Convert the labels.
        :param row: List[str]
        :param label_filter: xx
        :return: Dict
        """
        return {
            "study": row[0],
            "patient": row[1],
            "sex": row[2],
            "age": int(row[3]),
            "ImagerPixelSpacing_y": row[4],
            "PhotometricInterpretation": row[5],
            "gold": row[6],
            "val": row[7],
            "test": row[8],
            "test_2": row[9],
            "labels": self.convert_labels(row[6], row[9:], label_filter),
        }

    def convert_labels(self, gold_bool: bool, labels: torch.tensor, label_filter: List[int]) -> torch.tensor:
        labels = torch.tensor(labels)
        if type(label_filter[0]) == list:
            # check if label_filter contains two lists
            if gold_bool:
                labels = torch.tensor([labels[i] for i in label_filter[0]])  # get gold labels
            else:
                labels = torch.tensor([labels[i] for i in label_filter[1]])  # get silver labels
        else:
            if label_filter is not None:
                labels = torch.tensor([labels[i] for i in label_filter])
        return labels

    def read_label_csv(self, file, gold_filter, label_filter, cut_filter) -> Tuple[Dict, List[str]]:
        """
        Read UKB label csv.
        :param file: Path to .csv
        :param label_filter: xx
        :return: Tuple[Dict, List[str]] Label data dictionary and list of label names.
        """
        data: Dict[ImagePath, Dict] = {}
        label_file = pd.read_csv(file, sep=";")
        label_names = label_file.keys()[9:]
        # label_filter: List[int] = self.init_label_filter(label_filter, label_names)
        # logger.info(f"Found labels in {file}: {label_names}")
        for ind in tqdm(range(len(label_file)), desc=f"Reading label csv file {file}"):
            row = label_file.iloc[ind]
            image_path = os.path.join(self.base_path + "/PNG/", str(row[0]) + self.extension)
            size_kb = os.path.getsize(image_path) / 1000
            with open("/data/Helen/key2med-ai/damaged_imgs.pkl", "rb") as f:
                damaged_imgs = pickle.load(f)
            if image_path not in damaged_imgs:
                if size_kb > 600:
                    if gold_filter == "gold":
                        if row["gold"] == 1:
                            if cut_filter and (
                                row["QC_Bild_abgeschnitten_gold"] == 1 or row["QC_Aufnahme_verdreht_gold"] == 1
                            ):
                                continue
                            data[image_path] = self.read_row(row, label_filter)

                    elif gold_filter == "silver":
                        if row["gold"] == 0:
                            if cut_filter and (
                                row["QC_Bild_abgeschnitten_gold"] == 1 or row["QC_Aufnahme_verdreht_gold"] == 1
                            ):
                                continue
                            data[image_path] = self.read_row(row, label_filter)
                    else:
                        data[image_path] = self.read_row(row, label_filter)

        if label_filter is not None:
            if type(label_filter[0]) == list:
                label_filter = label_filter[0]
            label_names = [label_names[i] for i in label_filter]
        return data, label_names


class ChexpertLabelReader(LabelReader):
    sex_values = {"Male": 0, "Female": 1, "Unknown": 2}
    view_values = {"Frontal": 0, "Lateral": 1}
    direction_values = {"AP": 0, "PA": 1, "LL": 2, "RL": 3, "": 4}

    def __init__(
        self,
        label_file,
        label_filter: List[int] = None,
        uncertain_to_one: Union[str, List[str]] = None,
        uncertain_to_zero: Union[str, List[str]] = None,
        uncertain_upper_bound: float = 0.5,
        uncertain_lower_bound: float = 0.5,
        one_labels: List[str] = None,
        valid_views: List[str] = None,
        valid_sexs: List[str] = None,
        valid_directions: List[str] = None,
        min_age: int = None,
        max_age: int = None,
        fix_labels_by_hierarchy: bool = False,
        max_size: int = None,
    ):
        self.label_file = label_file
        self.max_size = max_size

        self.valid_views: List[str] = valid_views or list(self.view_values.keys())
        self.valid_directions: List[str] = valid_directions or list(self.direction_values.keys())
        self.valid_sexs: List[str] = valid_sexs or list(self.sex_values.keys())
        self.min_age = min_age or 0
        self.max_age = max_age or 100

        self.uncertain_to_one, self.uncertain_to_zero = self.init_uncertain_mapping(uncertain_to_one, uncertain_to_zero)
        self.uncertain_upper_bound, self.uncertain_lower_bound = uncertain_upper_bound, uncertain_lower_bound

        self.hierarchy = None
        if fix_labels_by_hierarchy:
            self.hierarchy = default_hierarchy

        self.data, self.index_to_label = self.read_label_csv(label_file, label_filter)
        self.label_to_index = {label: index for index, label in enumerate(self.index_to_label)}

        self.one_indices: List[int] = self.init_one_indices(one_labels)

        self.data = self.filter_data(self.data)
        self.image_paths = list(self.data.keys())

    def init_one_indices(self, one_labels: List[str]) -> List[int]:
        if not one_labels:
            return []
        return [self.label_to_index[label] for label in one_labels]

    def __getitem__(self, path):
        data = self.data[path]
        return self.prepare_data(data)

    @property
    def label_dim(self) -> Optional[int]:
        """
        Number of labels in the dataset.
        Default CheXpert 14, competition mode 5.
        :return: int Number of labels
        """
        return len(self.index_to_label)

    def filter_data(self, data: Dict[ImagePath, Dict]):
        filtered_data = {}

        for image_path, image_data in data.items():
            if image_data["view"] not in self.valid_views:
                continue
            if image_data["direction"] not in self.valid_directions:
                continue
            if image_data["sex"] not in self.valid_sexs:
                continue
            if not self.min_age <= image_data["age"] <= self.max_age:
                continue
            if self.one_indices and not any([data["labels"][label_index] == 1.0 for label_index in self.one_indices]):
                continue
            filtered_data[image_path] = image_data
        return filtered_data

    def prepare_data(self, data: Dict) -> torch.Tensor:
        # sex = self.sex_values[data['sex']]
        # view = self.view_values[data['view']]
        # direction = self.direction_values[data['direction']
        labels = self.convert_labels_live(data["labels"])
        return labels

    def convert_labels_live(self, labels: torch.tensor) -> torch.tensor:
        labels = torch.tensor(
            [self.convert_uncertain_labels(label, self.index_to_label[i]) for i, label in enumerate(labels)]
        )
        return labels

    def convert_uncertain_labels(self, label: torch.tensor, label_name: str):
        if label != -1.0:
            return label

        if label_name in self.uncertain_to_zero:
            return 0.0
        if label_name in self.uncertain_to_one:
            return 1.0
        if self.uncertain_upper_bound != self.uncertain_lower_bound:
            return random.uniform(self.uncertain_lower_bound, self.uncertain_upper_bound)
        return self.uncertain_lower_bound

    @staticmethod
    def init_uncertain_mapping(
        uncertain_to_one: Optional[Union[str, List[str]]], uncertain_to_zero: Optional[Union[str, List[str]]]
    ):
        if uncertain_to_one is None:
            uncertain_to_one = []
        elif uncertain_to_one == "best":
            uncertain_to_one = ["Edema", "Atelectasis"]

        if uncertain_to_zero is None:
            uncertain_to_zero = []
        elif uncertain_to_zero == "best":
            uncertain_to_zero = ["Cardiomegaly", "Consolidation", "Pleural Effusion"]

        return uncertain_to_one, uncertain_to_zero

    def init_label_filter(self, label_filter: Union[List[int], str], label_names: List[str]) -> Optional[List[int]]:
        if label_filter is None or label_filter == "full":
            return None
        if label_filter == "competition":
            return [2, 5, 6, 8, 10]
        if isinstance(label_filter, list) and isinstance(label_filter[0], int):
            return label_filter
        if isinstance(label_filter, list) and isinstance(label_filter[0], str):
            return [index for index, label_name in enumerate(label_names) if label_name in label_filter]
        raise NotImplementedError

    def read_label_csv(self, file, label_filter) -> Tuple[Dict, List[str]]:
        """
        Read CheXpert label csv.
        :param file: Path to .csv
        :param label_filter: xx
        :return: Tuple[Dict, List[str]] Label data dictionary and list of label names.
        """
        data: Dict[ImagePath, Dict] = {}
        with open(file, "r") as f:
            reader = csv.reader(f)
            label_names = next(reader)[5:]
            label_filter: List[int] = self.init_label_filter(label_filter, label_names)
            logger.info(f"Found labels in {file}: {label_names}")
            for index, row in tqdm(enumerate(reader), desc=f"Reading label csv file {file}"):
                image_path = row[0]
                data[image_path] = self.read_row(row, label_names, label_filter)
                if self.max_size is not None and index > self.max_size:
                    break
        if label_filter is not None:
            label_names = [label_names[i] for i in label_filter]
        return data, label_names

    def read_row(self, row: List, label_names: List[str], label_filter: List[int]) -> Dict[str, Any]:
        """
        Read a single row in the label csv. Convert the labels.
        :param row: List[str]
        :param label_filter: xx
        :return: Dict
        """
        return {
            "sex": row[1],
            "age": int(row[2]),
            "view": row[3],
            "direction": row[4],
            "labels": self.convert_labels(row[5:], label_names, label_filter),
        }

    def convert_labels(self, labels: List[str], label_names: List[str], label_filter: List[int]) -> torch.Tensor:
        """
        Label conversion while reading the label file.
        Only done once before training. No random transformations here.

        Possible labels:
        '1.0': positive
        '0.0': negative
        '-1.0': uncertain
        '': no mention

        :param labels: Labels from row of .csv file. As strings.
        :param label_filter: xx
        :return: torch.Tensor or list of floats. Initially converted labels.
        """
        convert = {"1.0": 1.0, "0.0": 0.0, "": 0.0, "-1.0": -1.0}
        labels = torch.FloatTensor([convert[x] for x in labels])
        if self.hierarchy is not None:
            labels = fix_vector(self.hierarchy, {label: index for index, label in enumerate(label_names)}, labels)
        if label_filter is not None:
            labels = labels[label_filter]
        return labels
