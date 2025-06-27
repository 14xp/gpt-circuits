"""
Example script demonstrating how to prepare integer sequence data for GPT training.

This example creates synthetic integer sequences that simulate a simple mathematical pattern
and shows how to use the integer data preparation pipeline.

Run this example:
    python -m data.integer_data.example_prepare
"""

import json
import numpy as np
import os
from pathlib import Path

from data.integer_data.prepare import prepare_integer_dataset


def generate_synthetic_sequences(num_sequences: int = 1000, vocab_size: int = 100, block_size: int = 128, bos_token: int = 0):
    """
    Generate synthetic integer sequences for demonstration.
    
    This creates sequences that follow a simple pattern:
    - Start with BOS token (0)
    - Follow with a sequence that has some mathematical relationship
    
    Args:
        num_sequences: Number of sequences to generate
        vocab_size: Size of vocabulary (max token value)
        block_size: Length of each sequence
        bos_token: Beginning of sequence token
    
    Returns:
        List of integer sequences
    """
    sequences = []
    
    for i in range(num_sequences):
        sequence = [bos_token]  # Start with BOS token
        
        # Generate the rest of the sequence with some pattern
        # Pattern: arithmetic progression with some noise
        start_val = np.random.randint(1, vocab_size // 4)
        step = np.random.randint(1, 5)
        
        for j in range(1, block_size):
            # Arithmetic progression with some random noise
            val = (start_val + step * j) % (vocab_size - 1) + 1  # Avoid 0 (reserved for BOS)
            # Add some random noise occasionally
            if np.random.random() < 0.1:
                val = np.random.randint(1, vocab_size)
            sequence.append(val)
        
        sequences.append(sequence)
    
    return sequences


def create_example_datasets():
    """Create example datasets in different formats."""
    
    # Set parameters
    vocab_size = 100
    block_size = 32  # Smaller for demonstration
    num_sequences = 200
    bos_token = 99 # Should be the largest reserved token
    
    # Generate synthetic data
    sequences = generate_synthetic_sequences(num_sequences, vocab_size, block_size, bos_token)
    
    # Get output directory
    output_dir = os.path.dirname(__file__)
    
    # Create example JSON file
    json_data = [{"sequence": seq} for seq in sequences]
    json_file = os.path.join(output_dir, "example_data.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Created example JSON file: {json_file}")
    
    # Create example numpy file
    npy_data = np.array(sequences)
    npy_file = os.path.join(output_dir, "example_data.npy")
    np.save(npy_file, npy_data)
    
    print(f"Created example NPY file: {npy_file}")
    
    return json_file, npy_file, vocab_size, block_size, bos_token


def run_example():
    """Run the complete example pipeline."""
    
    print("=== Integer Data Preparation Example ===")
    print()
    
    # Create example datasets
    json_file, npy_file, vocab_size, block_size, bos_token = create_example_datasets()
    
    # Prepare the JSON dataset
    print("Preparing JSON dataset...")
    prepare_integer_dataset(
        input_file=json_file,
        vocab_size=vocab_size,
        block_size=block_size,
        train_split=0.8,
        bos_token=bos_token,
        num_shards=1,
        output_dir=os.path.join(os.path.dirname(__file__), "example_json_output")
    )
    
    print()
    
    # Prepare the NPY dataset
    print("Preparing NPY dataset...")
    prepare_integer_dataset(
        input_file=npy_file,
        vocab_size=vocab_size,
        block_size=block_size,
        train_split=0.8,
        bos_token=bos_token,
        num_shards=1,
        output_dir=os.path.join(os.path.dirname(__file__), "example_npy_output")
    )
    
    print()
    print("=== Example Complete ===")
    print()
    print("The prepared datasets can now be used for training with configurations like:")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - block_size: {block_size}")
    print(f"  - Model config: integer_{vocab_size}_{block_size//4}x4 (or create custom)")
    print()
    print("To train a model:")
    print("  python -m training.gpt --config=<custom_config> --data_dir=data/integer_data/example_json_output")


if __name__ == "__main__":
    run_example()