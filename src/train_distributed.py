import signal
import sys
from torchinfo import summary
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from ifxAvian import Avian

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

    # Radar Configuration
    dev_config = Avian.DeviceConfig(
        sample_rate_Hz = 2000000,       # 1MHZ
        rx_mask = 7,                      # activate RX1 and RX3
        tx_mask = 1,                      # activate TX1
        if_gain_dB = 25,                  # gain of 33dB
        tx_power_level = 31,              # TX power level of 31
        start_frequency_Hz = 58.5e9,        # 60GHz 
        end_frequency_Hz = 62.5e9,        # 61.5GHz
        num_chirps_per_frame = 32,       # 128 chirps per frame
        num_samples_per_chirp = 64,       # 64 samples per chirp
        chirp_repetition_time_s = 0.0003, # 0.5ms
        frame_repetition_time_s = 1/33,   # 0.15s, frame_Rate = 6.667Hz
        mimo_mode = 'off'                 # MIMO disabled
    )
    radar_config = {'dev_config': dev_config, 
                    'num_rx_antennas': 3, 
                    'num_beams': 32,
                    'max_angle_degrees': 40}

    try:
        config, args = load_configs()

        # TODO: resume training if needed

        # Set random seed for reproducibility
        init_seeds(config.training.seed, rank)

        # Print configuration if main process
        if is_main_process():
            print("Starting Training...")
            print(f"Distributed training: {world_size} GPU(s), rank {rank}")
            if args.resume or args.resume_from:
                print(f"Resuming training from: {args.resume_from or 'latest checkpoint'}")
                print(f"Using original configuration from checkpoint")
            else:
                print(f"Run ID: {config.training.run_id}")                

        # Create model
        model = SimpleCNN(in_channels=config.data.input_channels, num_classes=config.data.output_classes)
        
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
        train_loader, val_loader, _ = get_dataloaders(
            config,
            radar_config=radar_config,  # Assuming radar_config is part of data config
            distributed=(world_size > 1),
            rank=rank,
            world_size=world_size
        )

        # Create and run Trainer
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

    finally:
        # Cleanup distributed training resources
        cleanup_distributed()
        print("Distributed training resources cleaned up.")

if __name__ == "__main__":
    main()