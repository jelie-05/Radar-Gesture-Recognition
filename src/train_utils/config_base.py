from dataclasses import dataclass
from typing import Optional

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_epochs: int = 0

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 100
    run_id: Optional[str] = None
    use_amp: bool = False
    gradient_clip: float = 1.0
    seed: int = 1
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    model: str = "simple"  # Options: 'simple', 'mid'

@dataclass
class DataConfig:
    dataset_path: str = ""
    input_channels: int = 2
    output_classes: int = 3
    pin_memory: bool = True
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

@dataclass
class ExperimentConfig:
    experiment_name: str = "radargesture"
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()