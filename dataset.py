import torch
import torchaudio
import pandas as pd
import os

class UrbanSound8KDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for the UrbanSound8K dataset.
    Handles loading audio files, resampling, and padding/truncating to a fixed length.
    """
    def __init__(self, annotations_file, audio_dir, folds=None, target_sample_rate=16000, max_duration_seconds=4):
        """
        Args:
            annotations_file (str): Path to the CSV file with annotations.
            audio_dir (str): Directory with all the audio files.
            folds (list, optional): A list of folds to include. If None, all folds are used. Defaults to None.
            target_sample_rate (int, optional): The sample rate to which all audio will be resampled. Defaults to 16000.
            max_duration_seconds (int, optional): The fixed duration for all audio clips in seconds.
                                                 Shorter clips are padded, longer ones are truncated. Defaults to 4.
        """
        self.annotations = pd.read_csv(annotations_file)
        if folds:
            # Filter the dataframe for the specified folds and reset the index
            self.annotations = self.annotations[self.annotations['fold'].isin(folds)].reset_index(drop=True)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        # Calculate the number of samples for the desired duration
        self.num_samples = target_sample_rate * max_duration_seconds

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Retrieves and processes a single audio sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The processed waveform as a 1D tensor.
        """
        audio_sample_path = self._get_audio_sample_path(index)
        
        try:
            # Load the audio file
            waveform, sample_rate = torchaudio.load(audio_sample_path)
            
            # Ensure waveform is mono
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) # Add channel dimension
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True) # Mix down to mono

            # Resample to the target sample rate if necessary
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)

            # Pad or truncate to the fixed number of samples
            if waveform.shape[1] < self.num_samples:
                # Pad with zeros if the waveform is shorter
                waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))
            else:
                # Truncate if the waveform is longer
                waveform = waveform[:, :self.num_samples]
            
            # Return the waveform, removing the channel dimension
            return waveform.squeeze(0)
        except Exception as e:
            # Handle errors in loading or processing, return a tensor of zeros
            print(f"Error loading or processing file {audio_sample_path}: {e}")
            return torch.zeros(self.num_samples)

    def _get_audio_sample_path(self, index):
        """Constructs the full path to an audio file."""
        fold = f"fold{self.annotations.iloc[index]['fold']}"
        file_name = self.annotations.iloc[index]['slice_file_name']
        return os.path.join(self.audio_dir, fold, file_name)

if __name__ == '__main__':
    # This block is for demonstrating and testing the dataset class.
    # It will run only when the script is executed directly.
    
    # This path assumes the UrbanSound8K dataset is in the project's root directory
    dataset_path = "UrbanSound8K"
    metadata_path = os.path.join(dataset_path, "UrbanSound8K.csv")
    audio_dir = dataset_path

    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        print("Please ensure the UrbanSound8K dataset is downloaded and placed in the project root.")
    else:
        # Check for XPU availability for hardware acceleration
        if torch.xpu.is_available():
            device = "xpu"
            print("Using XPU device")
        else:
            device = "cpu"
            print("Using CPU device")

        # Create dataset and dataloader for a specific fold (e.g., fold 1)
        print("\n--- Testing UrbanSound8KDataset ---")
        urbansound_dataset = UrbanSound8KDataset(annotations_file=metadata_path, audio_dir=audio_dir, folds=[1])
        dataloader = torch.utils.data.DataLoader(urbansound_dataset, batch_size=4, shuffle=True)

        print(f"Loaded {len(urbansound_dataset)} samples from fold 1.")

        # Iterate through a few batches to test the dataloader
        print("\n--- Testing DataLoader ---")
        for i, waveforms in enumerate(dataloader):
            print(f"Batch {i+1}:")
            print(f"  Waveforms shape: {waveforms.shape}")
            print(f"  Waveforms dtype: {waveforms.dtype}")
            if i == 2: # Stop after inspecting 3 batches
                break
