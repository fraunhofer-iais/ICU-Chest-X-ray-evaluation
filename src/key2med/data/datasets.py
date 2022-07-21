import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import coloredlogs
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from key2med.data.image_reader import ImageReader, PNGJPGImageReader
<<<<<<< HEAD
from key2med.data.label_reader import (
    ChexpertLabelReader,
    UKBLabelReader,
)
=======
from key2med.data.label_reader import ChexpertLabelReader, UKBLabelReader
>>>>>>> b363d27189d4a3c6efbae691018f6a2ea5322d5f
from key2med.utils.helper import hash_dict
from key2med.utils.plotting import text_histogram
from key2med.utils.transforms import BaseTransform, RandomAffineTransform, Transform

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ImagePath = str
Label = str

DatasetIndex = int
ImageIndex = int


class BaseDataset(Dataset):
    image_path_mapping: List[Any] = None
    image_data: Dict[ImagePath, Dict] = None
    image_reader: ImageReader = None
    random_transform: Transform = None

    def __getitem__(self, index) -> Tuple[List[torch.Tensor], List[Any]]:
        if index >= len(self):
            raise IndexError
        image_path = self.image_path_mapping[index]
        image = self.image_reader.load_image(image_path)
        if self.random_transform is not None:
            image = self.random_transform(image)
        image_data = self.load_image_data(image_path)
        return image, image_data

    def __len__(self):
        return len(self.image_path_mapping)

    def load_image_data(self, image_path):
        raise NotImplementedError


class CheXpertDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_config: str = "train_valid",
        transform: Transform = None,
        random_transform: Transform = None,
        upsample_label: str = None,
        downsample_label: str = None,
        max_size: int = None,
        print_stats: bool = True,
        plot_stats: bool = False,
        # label_reader kwargs
        label_filter: Union[List[int], str] = None,
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
        # image_reader kwargs
        use_cache: bool = False,
        in_memory: bool = True,
        # multiprocessing kwargs
        rank: int = 0,
        world_size: int = 1,
    ):
        self.split = split
        self.split_config = split_config
        parent_directory, chexpert_directory = self.split_data_path(data_path)
        label_file, sub_split = self.get_label_file(split, split_config)
        self.label_reader = self.init_label_reader(
            label_file=os.path.join(parent_directory, chexpert_directory, label_file),
            label_filter=label_filter,
            uncertain_to_one=uncertain_to_one,
            uncertain_to_zero=uncertain_to_zero,
            uncertain_upper_bound=uncertain_upper_bound,
            uncertain_lower_bound=uncertain_lower_bound,
            one_labels=one_labels,
            valid_views=valid_views,
            valid_sexs=valid_sexs,
            valid_directions=valid_directions,
            min_age=min_age,
            max_age=max_age,
            fix_labels_by_hierarchy=fix_labels_by_hierarchy,
            max_size=max_size,
        )
        self.image_paths = self.get_image_paths(self.label_reader)
        if sub_split is not None:
            self.image_paths = self.split_paths(self.image_paths, sub_split=sub_split, train_size=0.9)
        if max_size is not None:
            self.image_paths = self.image_paths[:max_size]

        transform = transform or self.default_transform
        self.random_transform = random_transform
        cache_config, cache_path = None, None
        if use_cache:
            cache_config = self.get_cache_config(data_path, label_file, transform)
            os.makedirs(os.path.join(data_path, "cache"), exist_ok=True)
            cache_path = os.path.join(data_path, "cache", hash_dict(cache_config))

        self.image_reader = self.init_image_reader(
            base_path=parent_directory,
            image_paths=self.image_paths,
            transform=transform,
            cache_config=cache_config,
            cache_path=cache_path,
            in_memory=in_memory,
        )

        if upsample_label is not None:
            if isinstance(upsample_label, List):
                for label in upsample_label:
                    self.image_paths = self.upsample_data(self.image_paths, label)
            else:
                self.image_paths = self.upsample_data(self.image_paths, upsample_label)
        if downsample_label is not None:
            if isinstance(downsample_label, List):
                for label in upsample_label:
                    self.image_paths = self.downsample_data(self.image_paths, label)
            else:
                self.image_paths = self.downsample_data(self.image_paths, downsample_label)

        self.world_size = world_size
        self.rank = rank
        if self.world_size > 1:
            gpu_split_size = int(len(self) / self.world_size)
            gpu_split_indices = list(range(self.rank * gpu_split_size, (self.rank + 1) * gpu_split_size))
            self.image_paths = [self.image_paths[i] for i in gpu_split_indices]

        self.image_path_mapping: List[ImagePath] = self.create_image_path_mapping()
        if print_stats:
            self.print_label_stats()
        if plot_stats:
            self.plot_label_stats()

    def split_data_path(self, data_path: str) -> Tuple[str, str]:
        split = Path(data_path).parts
        return str(Path(*split[:-1])), split[-1]

    def init_label_reader(self, **label_reader_kwargs):
        return ChexpertLabelReader(**label_reader_kwargs)

    def init_image_reader(self, **image_reader_kwargs):
        return PNGJPGImageReader(**image_reader_kwargs)

    def get_image_paths(self, label_reader):
        return label_reader.image_paths

    def __len__(self):
        return len(self.image_paths)

    def load_image_data(self, image_path):
        return self.load_image_labels(image_path)

    def load_image_labels(self, image_path):
        return self.label_reader[image_path]

    def load_image_metadata(self, image_path):
        return self.label_reader.data[image_path]

    def upsample_data(self, image_paths, upsample_label: str):
        positive_items = self.get_positive_items(image_paths, upsample_label)
        n_upsample = len(image_paths) - 2 * len(positive_items)
        new_data = random.choices(positive_items, k=n_upsample)
        image_paths += new_data
        return image_paths

    def get_positive_items(self, image_paths, label):
        label_index = self.label_reader.label_to_index[label]
        positive_values = []
        for item in image_paths:
            if self.label_reader.data[item]["labels"][label_index] == 1.0:
                positive_values.append(item)
        return positive_values

    def get_negative_items(self, image_paths, label):
        label_index = self.label_reader.label_to_index[label]
        positive_values = []
        for item in image_paths:
            if self.label_reader.data[item]["labels"][label_index] == 0.0:
                positive_values.append(item)
        return positive_values

    def downsample_data(self, image_paths, downsample_label: str):
        positive_items = self.get_positive_items(image_paths, downsample_label)
        negative_items = self.get_negative_items(image_paths, downsample_label)
        negative_items = random.choices(negative_items, k=len(positive_items))
        image_paths = positive_items + negative_items
        return image_paths

    @property
    def imratio(self) -> float:
        all_labels = self.all_labels_flat
        return (all_labels.sum() / len(all_labels)).item()

    @property
    def imratios(self) -> List[float]:
        stats = self.label_stats
        return [stat["positive_count"] for stat in stats.values()]

    @property
    def all_labels_flat(self) -> torch.tensor:
        return self.all_labels.view(-1)

    @property
    def index_to_label(self):
        return self.label_reader.index_to_label

    @property
    def all_labels(self) -> torch.tensor:
        labels: List[torch.Tensor] = [
            self.label_reader[path] for path in self.image_path_mapping
        ]  ##attention changed parent class for small test
        labels: torch.tensor = torch.stack(labels)
        return labels

    @property
    def label_stats(self) -> Dict[Label, Dict]:
        label_stats: Dict[Label, Dict] = {}
        all_datas = self.all_labels
        n = len(all_datas)
        positive_counts = (all_datas == 1.0).sum(axis=0)
        negative_counts = (all_datas == 0.0).sum(axis=0)
        uncertain_counts = ((all_datas != 0.0) & (all_datas != 1.0)).sum(axis=0)
        for label_name, pos, neg, unc in zip(self.index_to_label, positive_counts, negative_counts, uncertain_counts):
            label_stats[label_name] = {
                "positive_count": pos,
                "positive_ratio": pos / n,
                "negative_count": neg,
                "negative_ratio": neg / n,
                "uncertain_count": unc,
                "uncertain_ratio": unc / n,
            }
        return label_stats

    def create_image_path_mapping(self) -> List[ImagePath]:
        return self.image_paths

    def get_cache_config(self, data_path, label_file, transform):
        config = {"data_path": data_path, "label_file": label_file, "transform": transform.config}
        return config

    @property
    def all_image_paths(self):
        return

    def print_label_stats(self):
        """
        Plot stats on the distribution of labels and image metadata to the
        command line.
        :return: None
        """
        label_stats = self.label_stats
        imratio_message = (
            f'\n\t{"=" * 10} SPLIT {self.split} {"=" * 10}:\n'
            f"\tTotal images in split:  {len(self.image_paths):,}\n"
            f"\tTotal items in split:  {len(self.image_path_mapping):,}\n"
            f"\tTotal imratio in split: {self.imratio:.1%}.\n"
        )

        max_label_length: int = max([len(label) for label in label_stats.keys()])
        for label, stats in label_stats.items():
            imratio_message += (
                f"\t{label: <{max_label_length + 1}}: "
                f'{stats["positive_count"]:>7,} positive, '
                f'{stats["negative_count"]:>7,} negative, '
                f'{stats["uncertain_count"]:>7,} uncertain, '
                f'{stats["positive_ratio"]:>7.1%} imratio.\n'
            )
        logger.info(imratio_message)

    @property
    def all_metadata(self) -> List[Dict]:
        return [self.load_image_metadata(path) for path in self.image_path_mapping]

    def plot_label_stats(self):
        all_labels = self.all_labels_flat
        try:
            text_histogram(all_labels, title=f"distribution of all labels, split: {self.split}")
        except Exception as e:
            logger.info(f"Can not plot stats on data labels: {str(e)}")

        all_metadata = self.all_metadata
        all_sex_values = [data["sex"] for data in all_metadata]
        try:
            text_histogram(all_sex_values, title=f"distribution of sex values, split: {self.split}")
        except Exception as e:
            logger.info(f"Can not plot stats on sex values: {str(e)}")

        all_age_values = [data["age"] for data in all_metadata]
        try:
            text_histogram(all_age_values, title=f"distribution of age values, split: {self.split}")
        except Exception as e:
            logger.info(f"Can not plot stats on age values: {str(e)}")

        all_front_lateral_values = [data["view"] for data in all_metadata]
        try:
            text_histogram(all_front_lateral_values, title=f"distribution of front_lateral values, split: {self.split}")
        except Exception as e:
            logger.info(f"Can not plot stats on front_lateral values: {str(e)}")

        all_ap_pa_values = [data["direction"] for data in all_metadata]
        try:
            text_histogram(all_ap_pa_values, title=f"distribution of ap_pa values, split: {self.split}")
        except Exception as e:
            logger.info(f"Can not plot stats on ap_pa values: {str(e)}")

    @property
    def default_transform(self) -> Transform:
        return BaseTransform()

    @property
    def default_random_transform(self) -> Transform:
        return RandomAffineTransform()

    @property
    def image_dim(self):
        return self.image_reader.image_dim

    @property
    def label_dim(self):
        return self.label_reader.label_dim

    @staticmethod
    def split_paths(paths, sub_split, train_size: float = 0.9) -> List[ImagePath]:
        train_paths, test_paths = train_test_split(paths, train_size=train_size)
        if sub_split == "train":
            return train_paths
        if sub_split == "test":
            return test_paths

    def get_label_file(self, split, split_config):
        if split_config == "train_valid_test":
            if split == "train":
                return "train.csv", "train"
            if split == "valid":
                return "train.csv", "test"
            if split == "test":
                return "valid.csv", None
        elif split_config == "train_valid":
            if split == "train":
                return "train.csv", None
            if split == "valid":
                return "valid.csv", None
        raise NotImplementedError


class UKBDataset(CheXpertDataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_config: str = "train_valid",
        transform: Transform = None,
        random_transform: Transform = None,
        max_size: int = None,
        print_stats: bool = True,
        plot_stats: bool = False,
        upsample_label: str = None,
        downsample_label: str = None,
        gold_filter: str = None,
        label_filter: List[int] = None,
        cut_filter: bool = None,
        # image_reader kwargs
        use_cache: bool = False,
        in_memory: bool = True,
        # multiprocessing kwargs
        rank: int = 0,
        world_size: int = 1,
    ):
        self.split = split
        self.split_config = split_config

        extension = ".png"
        self.label_reader = UKBLabelReader(
            base_path=data_path,
            max_size=None,
            extension=extension,
            gold_filter=gold_filter,
            label_filter=label_filter,
            cut_filter=cut_filter,
        )
        self.image_paths = self.label_reader.image_paths
        self.image_paths = self.split_paths(self.image_paths, self.split)
        if max_size is not None:
            self.image_paths = self.image_paths[:max_size]
        test = self.load_image_metadata(self.image_paths[0])

        transform = transform or self.default_transform
        self.random_transform = random_transform
        cache_config, cache_path = None, None
        if use_cache:
            cache_config = self.get_cache_config(data_path, self.split, transform)
            os.makedirs(os.path.join(data_path, "cache"), exist_ok=True)
            cache_path = os.path.join(data_path, "cache", hash_dict(cache_config))
        self.image_path_mapping: List[ImagePath] = self.create_image_path_mapping()
        if print_stats:
            self.print_label_stats()
        if plot_stats:
            self.plot_label_stats()

        if "IntensivThorax" in data_path:
            self.image_reader = PNGJPGImageReader(
                base_path=data_path,
                image_paths=self.image_paths,
                transform=transform,
                cache_config=cache_config,
                cache_path=cache_path,
                in_memory=in_memory,
                label_reader=self.label_reader,
            )
        else:
            raise NotImplementedError

        if upsample_label is not None:
            self.image_paths = self.upsample_data(self.image_paths, upsample_label)
        if downsample_label is not None:
            self.image_paths = self.downsample_data(self.image_paths, downsample_label)

        self.world_size = world_size
        self.rank = rank
        if self.world_size > 1:
            gpu_split_size = int(len(self) / self.world_size)
            gpu_split_indices = list(range(self.rank * gpu_split_size, (self.rank + 1) * gpu_split_size))
            self.image_paths = [self.image_paths[i] for i in gpu_split_indices]

        if print_stats:
            self.print_label_stats()
        if plot_stats:
            self.plot_label_stats()

    def split_paths(self, paths, split):
        # load fixed test and validation sets
        img_paths_train = []
        patient_ids_train = []
        with open("/data/Hayal/key2med_fix/key2med-ai/patient_ids_valid_test.pkl", "rb") as handle:
            patient_ids = pickle.load(handle)
        with open("/data/Hayal/key2med_fix/key2med-ai/img_paths_val.pkl", "rb") as handle:
            img_paths_val = pickle.load(handle)
        with open("/data/Hayal/key2med_fix/key2med-ai/img_paths_test.pkl", "rb") as handle:
            img_paths_test = pickle.load(handle)
        with open("/data/Hayal/key2med_fix/key2med-ai/img_paths_test2.pkl", "rb") as handle:
            img_paths_test2 = pickle.load(handle)
        for key in paths:
            meta_data = self.label_reader[key]
            patient_id = meta_data["patient"]
            if meta_data["val"] != 1 and meta_data["test"] != 1:
                if patient_id not in patient_ids:
                    img_paths_train.append(key)
        if split == "train":
            return img_paths_train
        if split == "valid":
            return img_paths_val
        if split == "test":
            return img_paths_test
        if split == "test_2":
            return img_paths_test2
        raise NotImplementedError(f"Combination of split {split} does not work.")

    @property
    def default_transform(self) -> Transform:
        return BaseTransform()

    @property
    def all_labels(self) -> torch.tensor:
        labels: List[torch.tensor] = [self.label_reader[path]["labels"] for path in self.image_path_mapping]
        labels: torch.tensor = torch.stack(labels)

        return labels

    def load_image_labels(self, image_path):
        return self.label_reader[image_path]["labels"]


def main():
    pass
