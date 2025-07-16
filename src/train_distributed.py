import signal
import sys
from torchsummary import summary
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.train_utils.loader import get_dataloaders
from src.train_utils.distributed import (
    setup_distributed, 
    cleanup_distributed, 
    is_main_process,
    get_rank,
    get_world_size,
    init_seeds
)
from src.model.simple_model import SimpleCNN
from src.train_utils.utils import load_configs
from src.train_utils.trainer import Trainer


def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    cleanup_distributed()
    sys.exit(0)

def main():
    """Main training function."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup distributed training
    device, rank, world_size, local_rank = setup_distributed()

    try:
        config, args = load_configs()

        # TODO: resume training if needed

        # Set random seed for reproducibility
        init_seeds(config.training.seed, rank)

        # Print configuration if main process
        if is_main_process():
            print("Starting Training...")
            print(f"Distributed training: {world_size} GPU(s), rank {rank}")
            if args.resume_training or args.resume_from:
                print(f"Resuming training from: {args.resume_from or 'latest checkpoint'}")
                print(f"Using original configuration from checkpoint")
            else:
                print(f"Run ID: {config.training.run_id}")                

        # Create model
        model = SimpleCNN(in_channels=config.in_channels, num_classes=config.num_classes)
        
        # Print model summary if main process
        if is_main_process():
            try:
                summary(model, depth=8)
                print("Model created successfully!")
                print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
            except Exception as e:
                print(f"Could not print model summary: {e}")

        # Wrap model with DDP if distributed
        if world_size > 1:
            # Find unused parameters automatically
            model = DDP(
                model, 
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,  # All parameters are used
                gradient_as_bucket_view=True   # Optimize gradient memory layout
            )
            if is_main_process():
                print("Model wrapped with DistributedDataParallel")

        # Create dataloaders
        train_loader, val_loader, _ = get_dataloaders()

        # Create and run Trainer
        trainer = Trainer()

        train_stats = trainer.train()

    finally:
        # Cleanup distributed training resources
        cleanup_distributed()

if __name__ == "__main__":
    main()