import os
from natsort import natsorted
from torch.utils.data import Dataset
import soundfile as sf
import torch

class DataLoader(Dataset):
    """
    Custom DataLoader for loading and processing audio data.

    This class loads audio files from a given directory structure, applies optional transformations, 
    and handles both training and validation datasets. Additionally, it supports a split mode for 
    splitting the data without applying transformations.

    Attributes:
        val (bool): Indicates whether the dataset is for validation.
        split_mode (bool): Indicates whether to use split mode (no transformations).
        root (str): Root directory for the dataset (training or validation).
        label_encoding (dict): Dictionary for encoding labels.
        label (list): List of label names.
        label_detail (dict): Dictionary detailing the files for each label.
        data (list): List of dictionaries containing file paths, labels, types, and IDs.
        transforms (callable, optional): Transformations to be applied to the audio data.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns the sample at the given index.
    """

    def __init__(self, config: dict, transforms=None, val=False, split_mode=False):
        """
        Initializes the DataLoader with the given configuration.

        Args:
            config (dict): Configuration dictionary containing dataset paths and label encodings.
            transforms (callable, optional): Transformations to be applied to the audio data.
            val (bool): Indicates whether the dataset is for validation.
            split_mode (bool): Indicates whether to use split mode (no transformations).
        """
        self.val = val
        self.split_mode = split_mode
        if self.split_mode:
            print('You are in split mode. Transformation won\'t be applied.')
        if self.val:
            self.root = config["validation"]
        else:
            self.root = config["train"]

        self.label_encoding = config["label_encoding"]
        self.label = os.listdir(self.root)

        self.label_detail = {key: os.listdir(os.path.join(self.root, key)) for key in self.label}
        self.data = [
            {
                "path": os.path.join(self.root, label, val, files),
                "label": self.label_encoding[label],
                "type": val.split('_')[1],
                "Id": val.split('_')[-1]
            }
            for label, sample in self.label_detail.items()
                for val in sample
                    for files in os.listdir(os.path.join(self.root, label, val))
        ]
        self.transforms = transforms

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the sample at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing audio data, sampling rate, and other metadata.
        """
        data = self.data[index]
        if self.split_mode:
            try:
                data["audio"], data["sampling_rate"] = sf.read(data["path"])
                data["audio"] = torch.tensor(data["audio"])
            except:
                os.remove(data["path"])
                data["audio"] = None
                print(f'Removed - {data["path"]}')
        else:
            data["audio"], data["sampling_rate"] = sf.read(data["path"])
            data["audio"] = torch.tensor(data["audio"])

            if self.transforms:
                data["audio"] = self.transforms(data["audio"])

        return data