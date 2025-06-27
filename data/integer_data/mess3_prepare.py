"""
Mess3 script demonstrating how to prepare integer sequence data for GPT training.

This example creates synthetic integer sequences that simulate a simple mathematical pattern
and shows how to use the integer data preparation pipeline.

Run this example:
    python -m data.integer_data.mess3_prepare
"""

import json
import numpy as np
import os
from pathlib import Path

from data.integer_data.prepare import prepare_integer_dataset
from data.integer_data import mess3, sample_tokens


def create_mess3_datasets():
    """Create Mess3 datasets in different formats."""
    
    # Set parameters
    vocab_size = 4 # Mess3 + bos_token 
    block_size = 10  # Smaller for demonstration
    num_tokens = 20 * 1e6
    num_sequences = int(num_tokens // block_size)
    bos_token = 3 # Should be the largest reserved token
    seed = 42
    
    # Set up HMM data source
    x = 0.15  
    a = 0.6
    transition_tensor = mess3(x, a)
    initial_belief = np.array([1/3, 1/3, 1/3]) 

    # Sample sequences from the transition tensor
    sequences = sample_tokens(transition_matrix=transition_tensor, 
                              initial_belief=initial_belief, 
                              n_samples=num_sequences, 
                              n_tokens=block_size, 
                              seed=seed
                              )
    
    # Get output directory
    output_dir = os.path.dirname(__file__)
    
    # Create Mess3 JSON file
    json_data = [{"sequence": seq.tolist()} for seq in sequences]
    json_file = os.path.join(output_dir, "mess3_data.json")
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Created Mess3 JSON file: {json_file}")
    
    # Create Mess3 numpy file
    npy_data = np.array(sequences)
    npy_file = os.path.join(output_dir, "mess3_data.npy")
    np.save(npy_file, npy_data)
    
    print(f"Created Mess3 NPY file: {npy_file}")
    
    return json_file, npy_file, vocab_size, block_size, bos_token


def run_mess3():
    """Run the complete Mess3 pipeline."""
    
    print("=== Mess3 Integer Data Preparation Example ===")
    print()
    
    # Create Mess3 datasets
    json_file, npy_file, vocab_size, block_size, bos_token = create_mess3_datasets()
    
    # Prepare the JSON dataset
    print("Preparing Mess3 JSON dataset...")
    prepare_integer_dataset(
        input_file=json_file,
        vocab_size=vocab_size,
        block_size=block_size,
        train_split=0.8,
        bos_token=bos_token,
        num_shards=1,
        output_dir=os.path.join(os.path.dirname(__file__), "mess3_json_output")
    )
    
    print()
    
    # Prepare the NPY dataset
    print("Preparing Mess3 NPY dataset...")
    prepare_integer_dataset(
        input_file=npy_file,
        vocab_size=vocab_size,
        block_size=block_size,
        train_split=0.8,
        bos_token=bos_token,
        num_shards=1,
        output_dir=os.path.join(os.path.dirname(__file__), "mess3_npy_output")
    )
    
    print()
    print("=== Mess3 Example Complete ===")
    print()
    print("The prepared datasets can now be used for training with configurations like:")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - block_size: {block_size}")
    print(f"  - Model config: integer_{vocab_size}_{block_size//4}x4 (or create custom)")
    print()
    print("To train a model:")
    print("  python -m training.gpt --config=<custom_config> --data_dir=data/integer_data/mess3_json_output")


if __name__ == "__main__":
    run_mess3()