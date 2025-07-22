import signal
import sys
from torchinfo import summary
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

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
from src.train_utils.utils import load_config_from_args
from src.train_utils.trainer import Trainer


def signal_handler(signum, frame):
    """Handle termination signals for graceful shutdown."""
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    cleanup_distributed()
    sys.exit(0)

def load_checkpoint_config(checkpoint_path: str):
    """Load configuration from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            return checkpoint['config']
        else:
            print(f"Warning: No config found in checkpoint {checkpoint_path}")
            return None
    except Exception as e:
        print(f"Error loading config from checkpoint: {e}")
        return None

def main():
    """Main training function."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup distributed training
    device, rank, world_size, local_rank = setup_distributed()
    print(f"Training Device Setup: dev: {device}; rank: {rank}; world size: {world_size}; local rank: {local_rank}")

    # Rewrite dev config as dictionary
    dev_config = {
        'sample_rate_Hz': 2000000,
        'rx_mask': 7,
        'tx_mask': 1,
        'if_gain_dB': 25,
        'tx_power_level': 31,
        'start_frequency_Hz': 58.5e9,
        'end_frequency_Hz': 62.5e9,
        'num_chirps_per_frame': 32,
        'num_samples_per_chirp': 64,
        'chirp_repetition_time_s': 0.0003,
        'frame_repetition_time_s': 1/33,
        'mimo_mode': 'off'
    }  

    radar_config = {'dev_config': dev_config, 
                    'num_rx_antennas': 3, 
                    'num_beams': 32,
                    'max_angle_degrees': 40}

    try:
        config, args = load_config_from_args()

        # Handle resume options
        resume_training = getattr(args, 'resume', False)
        resume_from = getattr(args, 'resume_from', None)
        resume_run_id = getattr(args, 'resume_run_id', None)

        # If resuming
        if resume_training or resume_run_id:
            # If resuming from a specific run, find the latest checkpoint
            if resume_run_id and not resume_from:
                experiment_dir = Path(f"outputs/{config.experiment_name}")
                run_dir = experiment_dir / f"run_{resume_run_id}"
                print(f"Looking for run directory: {run_dir}")
                if run_dir.exists():
                    checkpoints_dir = run_dir / "checkpoints"
                    if checkpoints_dir.exists():
                        print(f"Found checkpoints directory: {checkpoints_dir}")
                        latest_checkpoint = checkpoints_dir / "latest.pth"
                        if latest_checkpoint.exists():
                            print(f"Resuming from latest checkpoint: {latest_checkpoint}")
                            resume_from = str(latest_checkpoint)
                            resume_training = True
                        else:
                            if is_main_process():
                                print(f"No checkpoint found in {checkpoints_dir}")
                            else:
                                if is_main_process():
                                    print(f"No checkpoints directory found for run {resume_run_id}")
                else:
                    if is_main_process():
                        print(f"Run {resume_run_id} not found")

        print(f"Resume training: {resume_training}, Resume from: {resume_from}, Run ID: {resume_run_id}")
        # Load configuration
        if resume_from:
            # Load config from checkpoint
            if is_main_process():
                print(f"Loading configuration from checkpoint: {resume_from}")
            config = load_checkpoint_config(resume_from)
            if config is None:
                if is_main_process():
                    print("Failed to load config from checkpoint, falling back to command line config")
                config, _ = load_config_from_args()
            else:
                if is_main_process():
                    print("Successfully loaded original configuration from checkpoint")
                # Apply any command line overrides
                if getattr(args, 'learning_rate', None) is not None:
                    config.training.optimizer.learning_rate = args.learning_rate
                    if is_main_process():
                        print(f"Overriding learning rate to {args.learning_rate}")
                if getattr(args, 'batch_size', None) is not None:
                    config.training.batch_size = args.batch_size
                    if is_main_process():
                        print(f"Overriding batch size to {args.batch_size}")
                if getattr(args, 'epochs', None) is not None:
                    config.training.epochs = args.epochs
                    if is_main_process():
                        print(f"Overriding epochs to {args.epochs}")
                if getattr(args, 'freeze', False):
                    config.model.encoder.freeze = True
                    if is_main_process():
                        print("Overriding encoder freeze to True")
        else:
            # Load config normally
            config, _ = load_config_from_args()

        # Extract run_id from checkpoint path if resuming
        run_id = getattr(args, 'run_id', None)
        if resume_from and not run_id:
            checkpoint_path = Path(resume_from)
            # Extract run_id from path (e.g., .../run_20250518_123045/checkpoints/latest.pth)
            run_dir = checkpoint_path.parent.parent
            if run_dir.name.startswith("run_"):
                run_id = run_dir.name[4:]  # Remove "run_" prefix
                if is_main_process():
                    print(f"Extracted run_id from checkpoint path: {run_id}")
        
        # Generate run_id if not resuming and not provided
        if not resume_training and run_id is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{timestamp}_rank{rank}"

        # Set random seed for reproducibility
        init_seeds(config.training.seed, rank)

        # Print configuration if main process
        if is_main_process():
            print("Starting Training...")
            print(f"Distributed training: {world_size} GPU(s), rank {rank}")
            if resume_training or resume_from:
                print(f"Resuming training from: {resume_from or 'latest checkpoint'}")
                print(f"Using original configuration from checkpoint")
            else:
                print(f"Run ID: {run_id}")                

        # Create model
        model = SimpleCNN(in_channels=config.data.input_channels, num_classes=config.data.output_classes).to(device=device)
        
        # Print model summary if main process
        if is_main_process():
            try:
                summary(model)
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
        print("=================================================================================")
        train_loader, val_loader, _ = get_dataloaders(
            config,
            radar_config=radar_config,  # Assuming radar_config is part of data config
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size
        )

        print(f"Dataloader is loaded successfully!")

        # Create and run Trainer
        print("=================================================================================")
        print("Training starts...")
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            run_id=config.training.run_id,
            rank=rank,
            world_size=world_size,
            resume_from=args.resume_from if args.resume else None,
            resume_training=args.resume
        )

        train_stats = trainer.train()

        # Print final results only on main process
        if is_main_process():
            print(f"Training complete!")
            # Print final training statistics

    finally:
        # Cleanup distributed training resources
        cleanup_distributed()
        print("Distributed training resources cleaned up.")

if __name__ == "__main__":
    main()