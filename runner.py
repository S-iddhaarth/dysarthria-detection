import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
class AudioMFCCDataset(Dataset):
    def __init__(self, audio_paths, sample_rate=16000, n_mfcc=13):
        """
        Dataset for converting audio files to MFCC
        
        Args:
            audio_paths (list): List of audio file paths
            sample_rate (int): Target sample rate
            n_mfcc (int): Number of MFCC coefficients
        """
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        """
        Load and convert single audio to MFCC
        
        Returns:
            torch.Tensor: MFCC representation
        """
        waveform, orig_sample_rate = torchaudio.load('/'.join(self.audio_paths[idx].split('\\')[1:]))
        
        # Resample if needed
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate, 
                new_freq=self.sample_rate
            ).to(self.device)
            waveform = resampler(waveform)
        
        # MFCC transformation
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            melkwargs={'n_fft': 400, 'hop_length': 160}
        ).to(self.device)
        
        mfccs = mfcc_transform(waveform.to(self.device))
        return mfccs.squeeze(0)  # Remove batch dimension

def dynamic_collate_fn(batch):
    """
    Custom collate function to handle variable-length MFCC sequences
    
    Args:
        batch (list): List of MFCC tensors
    
    Returns:
        torch.Tensor: Padded batch tensor
        torch.Tensor: Sequence lengths
    """
    # Sort batch by sequence length in descending order
    batch.sort(key=lambda x: x.shape[1], reverse=True)
    
    # Get sequence lengths
    lengths = torch.tensor([x.shape[1] for x in batch])
    
    # Pad sequences to max length
    max_length = batch[0].shape[1]
    padded_batch = torch.zeros(
        len(batch), 
        batch[0].shape[0], 
        max_length, 
        device=batch[0].device
    )
    
    for i, tensor in enumerate(batch):
        padded_batch[i, :, :tensor.shape[1]] = tensor
    
    return padded_batch, lengths

def convert_audio_to_mfcc(audio_files, batch_size=16):
    """
    Convert multiple audio files to MFCC batches
    
    Args:
        audio_files (list): List of audio file paths
        batch_size (int): Batch processing size
    
    Returns:
        tuple: Processed MFCC batches and their lengths
    """
    # Detect GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    dataset = AudioMFCCDataset(audio_files)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=dynamic_collate_fn,
    )
    
    # Store processed batches
    mfcc_batches = []
    mfcc_lengths = []
    
    for batch, lengths in dataloader:
        mfcc_batches.append(batch)
        mfcc_lengths.append(lengths)
    
    return mfcc_batches, mfcc_lengths

df = r'./data/UASPEECH/annotation.csv'
df = pd.read_csv(df)
audio_files = df["dysarthria"].to_list()
mfcc_batches, mfcc_lengths = convert_audio_to_mfcc(audio_files)

# Optionally, print batch and length information
for i, (batch, length) in enumerate(zip(mfcc_batches, mfcc_lengths)):
    print(f"Batch {i}: Shape {batch.shape}, Lengths {length}")