# Data loading pipeline for WSJ0-2mix/LibriMix datasets
# Updated to match MixClearNet paper requirements

import os
import soundfile as sf
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import random

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class WSJ0MixDataset(Dataset):
    """
    WSJ0-2mix dataset for speech separation as described in the MixClearNet paper
    """
    def __init__(self, dataset_path, target_length=4*16000, sample_rate=16000, num_speakers=2):
        self.dataset_path = dataset_path
        self.target_length = target_length  # 4 seconds as mentioned in paper
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers
        self.audio_files = []
        
        # Find all audio files and organize them
        self._build_file_list()
        
        if len(self.audio_files) == 0:
            logging.warning(f"No audio files found in {dataset_path}")
    
    def _build_file_list(self):
        """Build list of audio files"""
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    self.audio_files.append(os.path.join(root, file))
        
        logging.info(f"Found {len(self.audio_files)} audio files")
    
    def _create_mixture(self, audio1, audio2):
        """Create mixture of two speakers"""
        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Create mixture (equal power mixing as typical in WSJ0-2mix)
        mixture = audio1 + audio2
        
        return mixture, np.stack([audio1, audio2])
    
    def _pad_or_truncate(self, audio):
        """Pad or truncate audio to target length"""
        if len(audio) < self.target_length:
            # Pad with zeros
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            # Random crop for data augmentation
            start_idx = random.randint(0, len(audio) - self.target_length)
            audio = audio[start_idx:start_idx + self.target_length]
        
        return audio
    
    def __len__(self):
        # Use pairs of files for mixture creation
        return len(self.audio_files) // 2
    
    def __getitem__(self, idx):
        try:
            # Select two different audio files for mixing
            idx1 = idx * 2
            idx2 = min(idx * 2 + 1, len(self.audio_files) - 1)
            
            audio_path1 = self.audio_files[idx1]
            audio_path2 = self.audio_files[idx2]
            
            logging.debug(f"Loading files: {audio_path1}, {audio_path2}")
            
            # Load audio files
            audio1, sr1 = sf.read(audio_path1, dtype='float32')
            audio2, sr2 = sf.read(audio_path2, dtype='float32')
            
            # Ensure mono audio
            if audio1.ndim > 1:
                audio1 = audio1.mean(axis=1)
            if audio2.ndim > 1:
                audio2 = audio2.mean(axis=1)
            
            # Resample if necessary (in practice, WSJ0 is already 16kHz)
            if sr1 != self.sample_rate or sr2 != self.sample_rate:
                logging.warning(f"Sample rate mismatch: {sr1}, {sr2} vs {self.sample_rate}")
            
            # Normalize audio
            audio1 = audio1 / (np.max(np.abs(audio1)) + 1e-8)
            audio2 = audio2 / (np.max(np.abs(audio2)) + 1e-8)
            
            # Pad or truncate to target length
            audio1 = self._pad_or_truncate(audio1)
            audio2 = self._pad_or_truncate(audio2)
            
            # Create mixture
            mixture, clean_speakers = self._create_mixture(audio1, audio2)
            
            # Convert to torch tensors
            mixture = torch.from_numpy(mixture).float()
            clean_speakers = torch.from_numpy(clean_speakers).float()
            
            return mixture, clean_speakers
            
        except Exception as e:
            logging.error(f"Error loading audio at index {idx}: {e}")
            # Return dummy data if loading fails
            mixture = torch.zeros(self.target_length)
            clean_speakers = torch.zeros(self.num_speakers, self.target_length)
            return mixture, clean_speakers

class AudioDataset(Dataset):
    """
    Simplified audio dataset for basic loading (backward compatibility)
    """
    def __init__(self, dataset_path, target_length=16000):
        self.dataset_path = dataset_path
        self.audio_files = []
        self.target_length = target_length
        
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    self.audio_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        logging.debug(f"Attempting to load file: {audio_path}")
        
        try:
            waveform, sample_rate = sf.read(audio_path, dtype='float32')

            # Ensure mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Pad or truncate the waveform to the target length
            if len(waveform) < self.target_length:
                padding = self.target_length - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode='constant')
            else:
                waveform = waveform[:self.target_length]

            return torch.from_numpy(waveform).float(), sample_rate
            
        except Exception as e:
            logging.error(f"Error loading {audio_path}: {e}")
            # Return dummy data
            return torch.zeros(self.target_length), 16000

def load_dataset(dataset_name: str, batch_size: int = 8, use_mixtures: bool = True):
    """
    Load the specified dataset (WSJ0-2mix or LibriMix).
    
    Args:
        dataset_name: Name of the dataset subdirectory
        batch_size: Batch size for DataLoader
        use_mixtures: If True, use WSJ0MixDataset for speaker separation
    """
    dataset_path = os.path.join("wsj0", dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist.")

    print(f"Loading dataset from: {dataset_path}")
    
    if use_mixtures:
        # Use the WSJ0-2mix dataset for speech separation training
        dataset = WSJ0MixDataset(dataset_path, target_length=4*16000)  # 4 seconds
        print(f"Using WSJ0MixDataset with {len(dataset)} mixture pairs")
    else:
        # Use the simple dataset for backward compatibility
        dataset = AudioDataset(dataset_path)
        print(f"Using AudioDataset with {len(dataset)} files")
    
    # Create DataLoader with proper collate function
    def collate_fn(batch):
        if use_mixtures:
            mixtures, speakers = zip(*batch)
            mixtures = torch.stack(mixtures)
            speakers = torch.stack(speakers)
            return mixtures, speakers
        else:
            waveforms, sample_rates = zip(*batch)
            waveforms = torch.stack(waveforms)
            return waveforms, sample_rates[0]  # Assume same sample rate
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # For faster data loading
        pin_memory=True  # For faster GPU transfer
    )
    
    return dataloader

def create_wsj0_2mix_style_data(clean_dir: str, output_dir: str, num_mixtures: int = 1000):
    """
    Create WSJ0-2mix style data from clean speech files
    This is a utility function for data preparation
    """
    if not os.path.exists(clean_dir):
        print(f"Clean directory {clean_dir} does not exist")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all clean audio files
    clean_files = []
    for root, dirs, files in os.walk(clean_dir):
        for file in files:
            if file.endswith(".wav"):
                clean_files.append(os.path.join(root, file))
    
    if len(clean_files) < 2:
        print("Need at least 2 clean files to create mixtures")
        return
    
    print(f"Creating {num_mixtures} mixtures from {len(clean_files)} clean files")
    
    for i in range(num_mixtures):
        # Randomly select two different files
        file1, file2 = random.sample(clean_files, 2)
        
        try:
            # Load audio
            audio1, sr1 = sf.read(file1, dtype='float32')
            audio2, sr2 = sf.read(file2, dtype='float32')
            
            # Ensure mono
            if audio1.ndim > 1:
                audio1 = audio1.mean(axis=1)
            if audio2.ndim > 1:
                audio2 = audio2.mean(axis=1)
            
            # Ensure same length
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # Normalize
            audio1 = audio1 / (np.max(np.abs(audio1)) + 1e-8)
            audio2 = audio2 / (np.max(np.abs(audio2)) + 1e-8)
            
            # Create mixture
            mixture = audio1 + audio2
            
            # Save files
            sf.write(os.path.join(output_dir, f"mix_{i:04d}.wav"), mixture, sr1)
            sf.write(os.path.join(output_dir, f"s1_{i:04d}.wav"), audio1, sr1)
            sf.write(os.path.join(output_dir, f"s2_{i:04d}.wav"), audio2, sr1)
            
        except Exception as e:
            print(f"Error creating mixture {i}: {e}")
            continue
    
    print(f"Created {num_mixtures} mixtures in {output_dir}")

if __name__ == "__main__":
    # Test the dataset loading
    try:
        dataset_name = "si_dt_05"
        dataloader = load_dataset(dataset_name, batch_size=4, use_mixtures=True)
        print(f"Dataset loaded successfully!")
        
        # Test loading a batch
        for i, (mixtures, speakers) in enumerate(dataloader):
            print(f"Batch {i}: Mixture shape: {mixtures.shape}, Speakers shape: {speakers.shape}")
            if i >= 2:  # Just test first 3 batches
                break
                
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()