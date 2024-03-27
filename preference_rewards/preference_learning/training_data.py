import os
import torch
from torch.utils.data import Dataset, Sampler

class TrajectoryDataset(Dataset):
    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory, exist_ok=True)
        self.x_values, self.y_values, self.p_values = self.load()

    def add_entry(self, x_value, y_value, p_value):
        """Add a new entry to the dataset."""

        assert len(x_value.shape) == 2 and len(y_value.shape) == 1 and len(p_value.shape) == 1

        # Handle the case where this is the first entry
        if self.x_values is None or self.y_values is None or self.p_values is None:
            self.x_values = x_value.unsqueeze(0)
            self.y_values = y_value.unsqueeze(0)
            self.p_values = p_value.unsqueeze(0)
        else:
            # Append new entries to existing tensors
            self.x_values = torch.cat((self.x_values, x_value.unsqueeze(0)), 0)
            self.y_values = torch.cat((self.y_values, y_value.unsqueeze(0)), 0)
            self.p_values = torch.cat((self.p_values, p_value.unsqueeze(0)), 0)
        self.save()

    def load(self):
        """Load the dataset from a file."""
        path = os.path.join(self.directory, f"trajectories.pt")
        if os.path.isfile(path):
            loaded_data = torch.load(path)
            print(f"Loaded trajectory dataset from {path}")
            return loaded_data.get('x_values'), loaded_data.get('y_values'), loaded_data.get('p_values')
        else:
            # Return None for each component if the file doesn't exist
            return None, None, None

    def save(self):
        """Save the dataset to a file."""
        path = os.path.join(self.directory, f"trajectories.pt")
        torch.save({'x_values': self.x_values, 'y_values': self.y_values, 'p_values': self.p_values}, path)

    def __getitem__(self, index):
        """Fetch data by index."""
        # Return p_values alongside x_values and y_values
        return self.x_values[index], self.y_values[index], self.p_values[index]

    def __len__(self):
        """Return the total size of the dataset."""
        # The length is determined by the number of y_values or p_values
        return len(self.y_values) if self.y_values is not None else 0

    def get_data(self):
        """Get the entire dataset."""
        # Return p_values along with x_values and y_values
        return self.x_values, self.y_values, self.p_values

class BootstrapSampler(Sampler):
    """Sampler that draws samples with replacement."""
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        # Draw sample indices with replacement
        return iter(torch.randint(high=n, size=(n,), dtype=torch.int64).tolist())

    def __len__(self):
        return len(self.data_source)