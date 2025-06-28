#!/usr/bin/env python3
"""
Script to perform PCA analysis on GPT model activations.
Generates all possible length-N sequences with BOS token 3, captures activations,
and performs PCA visualization.
"""

import sys
import os
import itertools
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models.gpt import GPT


def generate_all_sequences() -> List[List[int]]:
    """
    Generate all possible length-10 sequences with BOS token 3.
    Format: [3, *, *, *, *, *, *, *, *, *] where * ∈ {0, 1, 2}
    Returns 3^block_size-1
    """
    print("Generating all possible sequences...")
    sequences = []
    block_size = 12  # Total length including BOS token
    
    # Generate all combinations of 9 positions with values {0, 1, 2}
    for combination in itertools.product([0, 1, 2], repeat=block_size-1):
        sequence = [3] + list(combination)  # BOS token 3 + 9 other tokens
        sequences.append(sequence)
    
    print(f"Generated {len(sequences)} sequences")
    return sequences


class ActivationCapture:
    """Helper class to capture activations during forward pass."""
    
    def __init__(self):
        self.intermediate_activations = []  # After attention, before MLP
        self.final_activations = []         # Before final layer norm
        
    def clear(self):
        self.intermediate_activations.clear()
        self.final_activations.clear()


def capture_activations_batch(model: GPT, sequences: List[List[int]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Capture activations for a batch of sequences.
    Returns intermediate, final, and post-layernorm activations.
    """
    batch_size = len(sequences)
    seq_len = len(sequences[0])
    
    # Convert sequences to tensor
    input_ids = torch.tensor(sequences, device=device)  # Shape: (batch_size, seq_len)
    
    intermediate_batch = []
    final_batch = []
    
    with torch.no_grad():
        # Manual forward pass to capture activations
        B, T = input_ids.size()
        
        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = model.transformer.wpe(pos)
        tok_emb = model.transformer.wte(input_ids)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in model.transformer.h:
            # Capture activation after attention but before MLP
            x_after_attn = x + block.attn(block.ln_1(x))
            intermediate_batch.append(x_after_attn.clone())
            
            # Continue through MLP
            x = x_after_attn + block.mlp(block.ln_2(x_after_attn))
        
        # x now contains final activations before layer norm
        final_batch.append(x.clone())
        
        # Apply final layer norm
        x_after_ln = model.transformer.ln_f(x)
        post_layernorm_batch = [x_after_ln.clone()]
    
    # Stack the captured activations
    intermediate_activations = intermediate_batch[0]  # Shape: (batch_size, seq_len, n_embd)
    final_activations = final_batch[0]                # Shape: (batch_size, seq_len, n_embd)
    post_layernorm_activations = post_layernorm_batch[0]  # Shape: (batch_size, seq_len, n_embd)
    
    return intermediate_activations, final_activations, post_layernorm_activations


def process_activations(activations: torch.Tensor) -> np.ndarray:
    """
    Process activations: remove BOS token and flatten over sequence dimension.
    Input shape: (num_sequences, seq_len, n_embd)
    Output shape: (num_sequences * (seq_len-1), n_embd)
    """
    block_size = 12

    # # Remove BOS token (position 0), keep positions 1-9
    # activations_no_bos = activations[:, 1:, :]  # Shape: (num_sequences, 9, n_embd)

    # Remove BOS token and final position (position 0 & final token), keep positions 1-8
    activations_no_bos = activations[:, 1:block_size-1, :]  # Shape: (num_sequences, 8, n_embd)

    # # Keep only final position (position 9)
    # activations_no_bos = activations[:, block_size-1:block_size, :]  # Shape: (num_sequences, 1, n_embd)
    
    # Flatten over sequence dimension
    num_sequences, seq_len_minus_1, n_embd = activations_no_bos.shape
    flattened = activations_no_bos.reshape(num_sequences * seq_len_minus_1, n_embd)
    
    return flattened.cpu().numpy()


def perform_pca_and_plot(intermediate_data: np.ndarray, final_data: np.ndarray, post_layernorm_data: np.ndarray):
    """
    Perform PCA on all three datasets and create visualization.
    """
    print("Performing PCA analysis...")
    
    # Perform PCA
    pca_intermediate = PCA(n_components=2)
    pca_final = PCA(n_components=2)
    pca_post_layernorm = PCA(n_components=2)
    
    intermediate_pca = pca_intermediate.fit_transform(intermediate_data)
    final_pca = pca_final.fit_transform(final_data)
    post_layernorm_pca = pca_post_layernorm.fit_transform(post_layernorm_data)
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Intermediate activations (after attention, before MLP)
    ax1.scatter(intermediate_pca[:, 0], intermediate_pca[:, 1], alpha=0.6, s=1)
    ax1.set_title(f'Intermediate Activations PCA\n(After Attention, Before MLP)')
    ax1.set_xlabel(f'PC1 ({pca_intermediate.explained_variance_ratio_[0]:.3f} variance)')
    ax1.set_ylabel(f'PC2 ({pca_intermediate.explained_variance_ratio_[1]:.3f} variance)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final activations (before final layer norm)
    ax2.scatter(final_pca[:, 0], final_pca[:, 1], alpha=0.6, s=1, color='orange')
    ax2.set_title(f'Final Activations PCA\n(Before Final Layer Norm)')
    ax2.set_xlabel(f'PC1 ({pca_final.explained_variance_ratio_[0]:.3f} variance)')
    ax2.set_ylabel(f'PC2 ({pca_final.explained_variance_ratio_[1]:.3f} variance)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Post-LayerNorm activations (after final layer norm, before unembedding)
    ax3.scatter(post_layernorm_pca[:, 0], post_layernorm_pca[:, 1], alpha=0.6, s=1, color='green')
    ax3.set_title(f'Post-LayerNorm Activations PCA\n(After Final Layer Norm)')
    ax3.set_xlabel(f'PC1 ({pca_post_layernorm.explained_variance_ratio_[0]:.3f} variance)')
    ax3.set_ylabel(f'PC2 ({pca_post_layernorm.explained_variance_ratio_[1]:.3f} variance)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'play/plots/pca_analysis_three_stages.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nPCA Results Summary:")
    print(f"Intermediate Activations (After Attention, Before MLP):")
    print(f"  Total variance explained by PC1+PC2: {sum(pca_intermediate.explained_variance_ratio_[:2]):.4f}")
    print(f"  PC1: {pca_intermediate.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca_intermediate.explained_variance_ratio_[1]:.4f}")
    
    print(f"\nFinal Activations (Before Final Layer Norm):")
    print(f"  Total variance explained by PC1+PC2: {sum(pca_final.explained_variance_ratio_[:2]):.4f}")
    print(f"  PC1: {pca_final.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca_final.explained_variance_ratio_[1]:.4f}")
    
    print(f"\nPost-LayerNorm Activations (After Final Layer Norm):")
    print(f"  Total variance explained by PC1+PC2: {sum(pca_post_layernorm.explained_variance_ratio_[:2]):.4f}")
    print(f"  PC1: {pca_post_layernorm.explained_variance_ratio_[0]:.4f}")
    print(f"  PC2: {pca_post_layernorm.explained_variance_ratio_[1]:.4f}")


def main():
    print("=== GPT Activation PCA Analysis ===\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model from checkpoints/mess3_12_2_64x1...")
    model_path = "checkpoints/mess3_12_2_64x1"
    model = GPT.load(model_path, device)
    model.eval()
    print(f"Model loaded successfully!")
    print(f"Config: {model.config}\n")
    
    # Generate all sequences
    sequences = generate_all_sequences()
    print(f"Total sequences to process: {len(sequences)}")
    print(f"Each sequence length: {len(sequences[0])}")
    print(f"Example sequence: {sequences[0]}\n")
    
    # Process sequences in batches to manage memory
    batch_size = 1000  # Adjust based on available memory
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    all_intermediate = []
    all_final = []
    all_post_layernorm = []
    
    print(f"Processing {num_batches} batches of size {batch_size}...")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        print(f"Processing batch {i+1}/{num_batches} (sequences {start_idx}-{end_idx-1})")
        
        # Capture activations for this batch
        intermediate_batch, final_batch, post_layernorm_batch = capture_activations_batch(model, batch_sequences, device)
        
        all_intermediate.append(intermediate_batch)
        all_final.append(final_batch)
        all_post_layernorm.append(post_layernorm_batch)
    
    # Concatenate all batches
    print("Concatenating all batches...")
    intermediate_activations = torch.cat(all_intermediate, dim=0)
    final_activations = torch.cat(all_final, dim=0)
    post_layernorm_activations = torch.cat(all_post_layernorm, dim=0)
    
    print(f"Intermediate activations shape: {intermediate_activations.shape}")
    print(f"Final activations shape: {final_activations.shape}")
    print(f"Post-LayerNorm activations shape: {post_layernorm_activations.shape}")
    
    # Process activations (remove BOS token and flatten)
    print("Processing activations...")
    intermediate_processed = process_activations(intermediate_activations)
    final_processed = process_activations(final_activations)
    post_layernorm_processed = process_activations(post_layernorm_activations)
    
    print(f"Processed intermediate shape: {intermediate_processed.shape}")
    print(f"Processed final shape: {final_processed.shape}")
    print(f"Processed post-LayerNorm shape: {post_layernorm_processed.shape}")
    print(f"Total activation vectors: {intermediate_processed.shape[0]}")
    
    # Perform PCA and create plots
    perform_pca_and_plot(intermediate_processed, final_processed, post_layernorm_processed)
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()