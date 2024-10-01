from pprint import pprint

import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
from torchinfo import summary


def describe_model(model):
    summary(model, input_size=(4, 5, 512, 512))  # input_data=np.stack([e['image'].shape for e in dataset])

def describe_predictions(predictions):
    batch=0
    print(f'{predictions.shape=}, printing only batch {batch}')
    for band in range(predictions.shape[0]):
        data = {
            'band' : band,
            'max' : [predictions[batch][band].max()],
            'min' : [predictions[batch][band].min()],
            'mean' : [predictions[batch][band].mean()],
            'std' : [predictions[batch][band].std()],
                }
        print(pd.DataFrame(data))


def describe_dataset(dataset, batch_size=4, show_batch=True):
    """
    Prints a detailed description of the dataset or subset, including:
    - Total number of samples
    - Sample at index 0 (its type, shape, and content if it's a dict)
    - Subset indices (if applicable)
    - First batch (if show_batch is True)

    Args:
        dataset (torch.utils.data.Dataset or torch.utils.data.Subset): The dataset to describe.
        batch_size (int): The batch size for testing if using a DataLoader.
        show_batch (bool): Whether to show the first batch of data.
    """
    print("Dataset Description".center(40, "-"))

    # Check if it's a Subset or a Dataset
    if isinstance(dataset, Subset):
        print("This is a Subset of the dataset.")
        print(f"Number of samples in the subset: {len(dataset)}")
        print(f"Indices used in the subset: {dataset.indices}")

        # Get the original dataset
        original_dataset = dataset.dataset
        print("\nOriginal Dataset:")
        print(f"Number of samples in the original dataset: {len(original_dataset)}")

    else:
        print(f"Number of samples in the dataset: {len(dataset)}")

    # Inspect a sample
    print("\nSample at index 0 (type and shape):")
    sample = dataset[0]

    # If the sample is a dictionary, display its keys and types of the values
    if isinstance(sample, dict):
        print(f"Sample is a dictionary with {len(sample)} keys:")
        for key, value in sample.items():
            value_type = type(value)
            value_shape = getattr(value, 'shape', 'N/A')
            print(f"  Key: '{key}' -> Type: {value_type} | Shape: {value_shape}")

            # If the value is a tensor or array, display more details
            if isinstance(value, (torch.Tensor, list, dict)):
                print(f"    Value contents (partial view): {str(value)[:200]}...")  # Truncate for large content

    else:
        pprint(f"Type: {type(sample)} | Shape: {getattr(sample, 'shape', 'N/A')}")

    # Show batch data if requested
    if show_batch:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch = next(iter(dataloader))

        print("\nFirst batch (batch_size = {}):".format(batch_size))

        # If it's a batch of (inputs, labels), display both
        if isinstance(batch, tuple):
            for i, part in enumerate(batch):
                print(f"Batch part {i}:")
                pprint(f"Type: {type(part)} | Shape: {getattr(part, 'shape', 'N/A')}")
        elif isinstance(batch, dict):
            print("Batch is a dictionary:")
            for key, value in batch.items():
                value_type = type(value)
                value_shape = getattr(value, 'shape', 'N/A')
                print(f"  Key: '{key}' -> Type: {value_type} | Shape: {value_shape}")
        else:
            pprint(f"Type: {type(batch)} | Shape: {getattr(batch, 'shape', 'N/A')}")

    print("-" * 40)

# Example usage:
# describe(dataset, batch_size=4, show_batch=True)
