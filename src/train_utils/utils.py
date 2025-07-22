import argparse
import yaml
from dataclasses import is_dataclass, fields
from src.train_utils.config_base import ExperimentConfig
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Radar Gesture Recognition")
    
    # Add arguments for model configuration
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('--local-rank', default=-1, type=int, help='Local rank for distributed training')

    # Add arguments for training configuration
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer')
    parser.add_argument('--run_id', type=str, 
                        help='Name for saving the model. If not provided, a timestamp will be used')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision for training')
    parser.add_argument('--gradient_clip', type=float, help='Gradient clipping value', default=1.0)
    
    # Resume training options
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to the checkpoint to resume from')
    parser.add_argument('--resume_run_id', type=str, default=None, help='Run ID to resume training from')
    
    args = parser.parse_args()

    return args

def load_from_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    
def dict_to_class(cls, d):
    if not is_dataclass(cls):
        return d
    field_types = {f.name: f.type for f in fields(cls)}
    return cls(**{
        k: dict_to_class(field_types[k], v) if k in field_types else v
        for k, v in d.items()
    })

def load_config_from_args():
    """
    Load configuration from YAML file and command line arguments.
    Returns:
        config (dict): Configuration dictionary.
        args (argparse.Namespace): Parsed command line arguments."""

    args = parse_args()
    config = load_from_yaml(args.config)
    config = dict_to_class(ExperimentConfig, config)

    # Override config with command line arguments if provided
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.learning_rate:
        config.training.optimizer.learning_rate = args.learning_rate
    if args.run_id:
        config.training.run_id = args.run_id
    if args.use_amp:
        config.training.use_amp = args.use_amp
    if args.gradient_clip:
        config.training.gradient_clip = args.gradient_clip
    if args.num_workers:
        config.training.num_workers = args.num_workers

    return config, args


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_confusion_elements(y_true, y_pred, n_classes):
    TP = torch.zeros(n_classes, dtype=torch.int32)
    FP = torch.zeros(n_classes, dtype=torch.int32)
    TN = torch.zeros(n_classes, dtype=torch.int32)
    FN = torch.zeros(n_classes, dtype=torch.int32)

    for c in range(n_classes):
        tp = (y_pred == c) & (y_true == c)
        fp = (y_pred == c) & (y_true != c)
        tn = (y_pred != c) & (y_true != c)
        fn = (y_pred != c) & (y_true == c)

        TP[c] = tp.sum()
        FP[c] = fp.sum()
        TN[c] = tn.sum()
        FN[c] = fn.sum()

    return TP, FP, TN, FN