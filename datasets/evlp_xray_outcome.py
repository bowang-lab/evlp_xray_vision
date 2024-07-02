import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from typing import Optional, Union, List


class OutcomeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        resolution,
        label_type,
        trend,
        batch_size=16,
        return_label=True,
        grayscale=False,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
    ):
        super(OutcomeDataModule, self).__init__()
        self.data_dir = data_dir
        self.labels_df = pd.read_csv(
            os.path.join(data_dir, "labels.csv"), index_col="EVLP_ID"
        )
        self.label_type = label_type
        self.trend = trend
        self.resolution = resolution
        self.batch_size = batch_size
        self.return_label = return_label
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.grayscale = grayscale

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.num_labels = 3 if label_type == "multiclass" else 1

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = OutcomeDataset(
                self.data_dir,
                split="train",
                label_type=self.label_type,
                trend=self.trend,
                resolution=self.resolution,
                return_label=self.return_label,
                grayscale=self.grayscale,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
            self.dataset_val = OutcomeDataset(
                self.data_dir,
                split="val",
                label_type=self.label_type,
                trend=self.trend,
                resolution=self.resolution,
                return_label=self.return_label,
                grayscale=self.grayscale,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )
        if stage == "test" or stage is None:
            self.dataset_test = OutcomeDataset(
                self.data_dir,
                split="test",
                label_type=self.label_type,
                trend=self.trend,
                resolution=self.resolution,
                return_label=self.return_label,
                grayscale=self.grayscale,
                norm_mean=self.norm_mean,
                norm_std=self.norm_std,
            )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=3
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=3
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=3
        )


class OutcomeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        resolution,
        label_type,
        trend,
        return_label=True,
        grayscale=False,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
    ):
        super(OutcomeDataset, self).__init__()
        assert split in ["train", "val", "test"]

        self.split = split
        self.data_dir = data_dir
        self.image_files = os.listdir(os.path.join(data_dir, split))
        self.labels_df = pd.read_csv(
            os.path.join(data_dir, "labels.csv"), index_col="EVLP_ID"
        )
        self.label_type = label_type
        self.resolution = resolution
        self.trend = trend
        self.num_labels = self.labels_df["Outcome"].nunique()
        self.return_label = return_label
        self.grayscale = grayscale

        if self.trend:
            image_files_1hr = []
            image_files_3hr = []

            for image_file in self.image_files:
                timepoint = image_file.split("_")[1]

                if "1h" in timepoint:
                    image_files_1hr.append(image_file)
                elif "3h" in timepoint:
                    image_files_3hr.append(image_file)

            image_files_1hr.sort()
            image_files_3hr.sort()
            filtered_image_files_1hr = []
            i = 0

            # The following for loop eliminates cases without both 1h and 3h images, and also ensures that the 1hr and 3hr images are paired correctly
            for image_file_3hr in image_files_3hr:
                image_name_3hr = image_file_3hr.split("_")[0]
                while True:
                    if i == len(image_files_1hr):
                        assert False, f"No 1hr image found for {image_name_3hr}"
                    image_name_1hr = image_files_1hr[i].split("_")[0]
                    if image_name_1hr == image_name_3hr:
                        filtered_image_files_1hr.append(image_files_1hr[i])
                        i += 1
                        break
                    else:
                        i += 1

            self.image_files = filtered_image_files_1hr
            self.image_files_3hr = image_files_3hr

            assert len(self.image_files) == len(self.image_files_3hr)  # Sanity check

        if self.label_type == "outcome":
            filtered_image_files = []
            filtered_image_files_3hr = [] if self.trend else None
            for i, image_file in enumerate(self.image_files):
                evlp_id = int(image_file.split("_")[0][4:])
                if self.labels_df.loc[evlp_id, "Outcome"] != 2:
                    filtered_image_files.append(image_file)
                    if self.trend:
                        image_file_3hr = self.image_files_3hr[i]
                        filtered_image_files_3hr.append(image_file_3hr)
            self.image_files = filtered_image_files
            if self.trend:
                self.image_files_3hr = filtered_image_files_3hr

        # Image augmentation
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.25, contrast=0.25),
                    transforms.RandomAffine(
                        degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        # Get the image tensor (i.e. load it, transform it, turn it into a tensor)
        image_file = self.image_files[item]
        image_path = os.path.join(self.data_dir, self.split, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        if self.grayscale:
            image = image.mean(dim=0, keepdim=True)

        if self.trend:
            image_file_3hr = self.image_files_3hr[item]
            image_path_3hr = os.path.join(self.data_dir, self.split, image_file_3hr)
            image_3hr = Image.open(image_path_3hr).convert("RGB")
            image_3hr = self.transform(image_3hr)
            if self.grayscale:
                image_3hr = image_3hr.mean(dim=0, keepdim=True)
            image = (image, image_3hr)

        # Get the label tensor
        evlp_id = int(
            image_file.split("_")[0][4:]
        )  # Remove the file extension and EVLP prefix
        label = self.labels_df.loc[evlp_id, "Outcome"]
        if self.label_type == "transplant":
            label = 1.0 if label == 2 else 0.0
        elif self.label_type == "outcome":
            label = 1.0 if label == 1 else 0.0
        else:
            label = int(label)
        label = torch.tensor(label)

        if self.return_label:
            return image, label
        else:
            return image

    def get_ids_and_labels(self):  # Used for data inference
        ids, labels = [], []
        for image_file in self.image_files:
            evlp_id = int(image_file.split("_")[0][4:])
            ids.append(evlp_id)
            labels.append(self.labels_df.loc[evlp_id, "Outcome"])
        return ids, labels

    def get_original_image(self, item):  # Used for saliency mapping
        image_file = self.image_files[item]
        image_path = os.path.join(self.data_dir, self.split, image_file)
        image = Image.open(image_path).convert("RGB")
        image_resized = image.resize((self.resolution, self.resolution))

        if self.trend:
            image_file_3hr = self.image_files_3hr[item]
            image_path_3hr = os.path.join(self.data_dir, self.split, image_file_3hr)
            image_3hr = Image.open(image_path_3hr).convert("RGB")
            image_3hr_resized = image_3hr.resize((self.resolution, self.resolution))
            image = (image, image_3hr)
            image_resized = (image_resized, image_3hr_resized)

        return image, image_resized


def test_dataloader():
    train_dataloader = OutcomeDataModule(
        data_dir=...,
        resolution=224,
        label_type="transplant",
        batch_size=2,
    )
    train_dataloader.setup(stage="fit")
    train_dataloader = train_dataloader.train_dataloader()
    it = iter(train_dataloader)
    for _ in range(10):
        train_features, train_labels = next(it)

    dataset = OutcomeDataset(
        data_dir=...,
        split="val",
        resolution=224,
        label_type="multiclass",
        trend=True,
    )
    ids, labels = dataset.get_ids_and_labels()
    image_files = [img for i, img in enumerate(image_files) if np.isnan(labels[i])]
    print(len(labels))
