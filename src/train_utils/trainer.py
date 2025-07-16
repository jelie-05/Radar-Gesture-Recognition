import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from torch.cuda.amp import GradScaler
import os
from pathlib import Path

from src.train_utils.distributed import (
    setup_distributed, 
    cleanup_distributed, 
    is_main_process,
    get_rank,
    get_world_size,
    init_seeds
)


class Trainer():
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        run_id: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        resume_from: Optional[str] = None,
        resume_training: bool = False,
    ):
        """
        Initialize the Trainer class.
        
        Args:
            model: Model to be trained.
            config: Configuration dictionary containing training parameters.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            test_loader: DataLoader for test data.
            device: Device to run the model on (CPU or GPU).
            run_id: Unique identifier for the training run.
            rank: Rank of the current process in distributed training.
            world_size: Total number of processes in distributed training.
            resume_from: Path to the checkpoint to resume training from.
            resume_training: Whether to resume training from the last checkpoint.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.run_id = run_id
        self.rank = rank
        self.world_size = world_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.use_amp = config.training.use_amp if 'use_amp' in config.training else False
        self.scaler = GradScaler() if self.use_amp and torch.cuda.is_available() else None

        if is_main_process() and self.use_amp:
            print(f"Using Automatic Mixed Precision (AMP)")
            if not torch.cuda.is_available():
                print("Warning: AMP enabled but CUDA not available, falling back to FP32")

        # Create base experiment directory (only on main process)
        if is_main_process():
            self.experiment_dir = Path(f"outputs/{config.experiment_name}")
            os.makedirs(self.experiment_dir, exist_ok=True)
        else:
            self.experiment_dir = Path(f"outputs/{config.experiment_name}")