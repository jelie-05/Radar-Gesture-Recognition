import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Radar Gesture Recognition")
    
    # Add arguments for model configuration
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)

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

def load_from_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_configs():
    """
    Load configuration from YAML file and command line arguments.
    Returns:
        config (dict): Configuration dictionary.
        args (argparse.Namespace): Parsed command line arguments."""

    args = parse_args()
    config = load_from_yaml(args.config)

    # Override config with command line arguments if provided
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['optimizer']['learning_rate'] = args.learning_rate
    if args.run_id:
        config['training']['run_id'] = args.run_id
    if args.use_amp:
        config['training']['use_amp'] = args.use_amp
    if args.gradient_clip:
        config['training']['gradient_clip'] = args.gradient_clip
    if args.num_workers:
        config['training']['num_workers'] = args.num_workers

    return config, args