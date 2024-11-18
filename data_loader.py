import os
from natsort import natsorted
from torch.utils.data import Dataset
import soundfile as sf
import torch
# from utils.utility import csv_to_tuples
import numpy as np
import pandas as pd
import torch.nn.functional as F
import wandb
from utils.utility import  csv_to_tuples
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

class SpeechPairLoader(Dataset):
    def __init__(self,path,select) -> None:
        data = pd.read_csv(path)
        unique_values = ['Very low (15%)', 'Low (29%)', 'Mid (58%)', 'High (86%)']
        self.set = unique_values[select]
        self.data = data[data['intt'] == self.set]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        series = self.data.iloc[index]
        # dys,cont,word,intt,mic = series[]
        dyss,dysd = sf.read(dys[3:])
        conts,contd = sf.read(cont[3:])
        
        return dyss,dysd,conts,contd,word,intt,mic

class SpeechPairLoaderPreprocessed(Dataset):
    def __init__(self,root,img_dir,img_type,annotation,select):
        self.root = root
        self.img_dir = img_dir
        self.type = img_type
        data = pd.read_csv(os.path.join(root,annotation))
        unique_values = ['Very low (15%)', 'Low (29%)', 'Mid (58%)', 'High (86%)']
        self.set = unique_values[select]
        self.data = data[data['intt'] == self.set]
        self.dys = "dysarthric_speech_path"
        self.cont = "controlled_speech_path"
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        series = self.data.iloc[index]
        cont_c = '/'.join(series[self.cont].split('\\'))
        dys_c = '/'.join(series[self.dys].split('\\'))
        cont_path = os.path.join(self.root,self.img_dir,self.type,cont_c)
        dys_path = os.path.join(self.root,self.img_dir,self.type,dys_c)
        cont = torch.tensor(np.load(cont_path),dtype=torch.float)
        dys = torch.tensor(np.load(dys_path),dtype=torch.float)
        # resized_cont = F.interpolate(cont.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        # resized_dys = F.interpolate(dys.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        return dys,cont