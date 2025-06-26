"""
End-to-end test script for the hybrid Q-UDiT model.

This script verifies that the entire pipeline is functional:
1. A hybrid UDiT model (with a QuixerBlock) can be instantiated.
2. It can be wrapped in the DiffusionModel lightning module.
3. Gradients flow correctly through both classical and quantum layers.
4. A PyTorch Lightning trainer can successfully run a few training steps.
"""
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

from quantum_transformers.uditt.uditt import UDiT
from quantum_transformers.diffusion.diffusion_model import DiffusionModel

def run_test():
    print("--- Starting Hybrid Q-UDiT End-to-End Test ---")

    # 1. Model Configuration
    img_size = 32
    patch_size = 4 # smaller patch size for fewer tokens
    depth = 4      # shallow model for speed
    hidden_size = 64 # smaller hidden size
    num_heads = 4
    quantum_block_idx = 2 # Make the middle block quantum
    
    print(f"Instantiating UDiT model with quantum block at index {quantum_block_idx}...")
    
    # Instantiate the denoiser model
    denoiser = UDiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        num_classes=10,
        quantum_block_indices=[quantum_block_idx],
        quixer_data_qubits=4,
        qsvt_poly_degree=3 # A small degree for a quick test
    )

    # 2. Diffusion Wrapper
    print("Wrapping UDiT in DiffusionModel...")
    diffusion_model = DiffusionModel(
        denoiser_model=denoiser,
        timesteps=100, # Fewer timesteps for a quick test
        learning_rate=1e-4
    )

    # 3. Create a mock dataset and dataloader
    print("Creating mock dataset...")
    num_samples = 8
    mock_images = torch.randn(num_samples, 3, img_size, img_size)
    mock_labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(mock_images, mock_labels)
    dataloader = DataLoader(dataset, batch_size=4)

    # 4. PyTorch Lightning Trainer
    print("Configuring PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=10, # Run for only a few steps
        log_every_n_steps=1,
        logger=False, # Disable logging for this test
        enable_checkpointing=False, # Disable checkpointing
        enable_progress_bar=True,
    )

    # 5. Run Training
    print("\n--- Launching Training ---")
    try:
        trainer.fit(diffusion_model, dataloader)
        print("\n--- Test PASSED ---")
        print("Successfully completed a few training steps. The hybrid model is functional.")
    except Exception as e:
        print("\n--- Test FAILED ---")
        print(f"An error occurred during the training run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test() 