import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from torch.cuda.amp import GradScaler
import os
from pathlib import Path
from datetime import datetime
import yaml
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from torch.cuda.amp import autocast
from src.train_utils.utils import AverageMeter, get_confusion_elements

from src.train_utils.distributed import (
    is_distributed, 
    get_rank, 
    get_world_size, 
    is_main_process,
    reduce_dict,
    save_on_master,
    synchronize,
    gather_tensor
)


class Trainer():
    def __init__(
        self,
        model: nn.Module,
        config,
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

        self.use_amp = getattr(self.config.training, 'use_amp', False)
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

        # Resume or initialize training
        self.start_epoch = 0
        self.resume_ckpt = None

        if resume_from or resume_training:
            pass

        # Generate run ID if not provided
        if self.run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{timestamp}"
            if is_main_process():
                print(f"Run ID is not configured. Generated run ID: {self.run_id}")

        # Create run directory
        if is_main_process():
            self.output_dir = self.experiment_dir / f"run_{self.run_id}"
            if not self.resume_ckpt and self.output_dir.exists():
                print(f"Run directory {self.output_dir} already exists. Adding unique suffix.")
                import uuid
                self.output_dir = self.experiment_dir / f"run_{self.run_id}_{uuid.uuid4().hex[:6]}"

            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Run directory created: {self.output_dir}")
        else:
            self.output_dir = self.experiment_dir / f"run_{self.run_id}"

        # Synchronize all processes before continuing
        if is_distributed():
            synchronize()

        # Set up optimizer
        if train_loader is not None:
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize training state variables
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_metrics = {}
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            # 'train_accuracy': [],
            # 'val_accuracy': [],
            'learning_rate': [],
            # 'train_precision': [],
            # 'val_precision': [],
            # 'train_recall': [],
            # 'val_recall': [],
        }

        # Resume from checkpoint if specified
        if self.resume_ckpt:
            self._load_ckpt(self.resume_ckpt)
        else:
            if is_main_process():
                self.save_run_config(config)

        # Initialize TensorBoard writer
        self.writer = None
        if is_main_process():
            tensorboard_dir = self.output_dir / "tensorboard"
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    
    def train(self):
        if is_main_process():
            print(f"Starting from epoch {self.start_epoch + 1} of {self.config.training.epochs} total epochs")
        
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.training.epochs):
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Training step
            train_metrics = self._train_epoch(epoch)

            if is_main_process():
                self.metrics_history['train_loss'].append(train_metrics['train_loss'])
                lr = self.optimizer.param_groups[0]['lr'] 
                self.metrics_history['learning_rate'].append(lr)

            # Validation step
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch(epoch)

                if is_main_process():
                    self.metrics_history['val_loss'].append(val_metrics['val_loss'])

                    # TODO: Evaluate best epoch

            # Update Scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Self regular checkpoint every 10 epochs or at the end of training
            if (epoch + 1) % 10 == 0 or epoch == self.config.training.epochs - 1:
                self.save_checkpoint(epoch, metrics=val_metrics)

        # Save metrics history
        # TODO: implement

        # Calculate training time
        total_time = time.time() - start_time

        # Return final stats
        return {
            'total_epochs': self.config.training.epochs,
            'starting_epoch': self.start_epoch,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'training_time': total_time,
            'metrics_history': self.metrics_history,
        }

    def _train_epoch(self, epoch):            
        self.model.train()

        # Log metrics
        loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cuda_mem = AverageMeter()
        TP_sum = 0
        FP_sum = 0
        TN_sum = 0
        FN_sum = 0

        start = time.time()

        # Only show progress bar if main process
        if is_main_process():
            pbar = tqdm(self.train_loader)
            pbar.set_description(f'[TRAINING] Epoch {epoch + 1}')
        else:
            pbar = self.train_loader

        total_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(pbar):
            # Measure dataloading time
            data_time.update(time.time() - start)

            inputs = batch['time_proj'].to(self.device)
            labels = batch['class'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # Scale loss to prevent gradient underflow
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()

                # Gradient clipping
                if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                    # Unscale before clipping to get true gradient norms
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_val
                    )

                # Update parameters
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without AMP
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if hasattr(self.config.training, 'grad_clip_val') and self.config.training.grad_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.grad_clip_val
                    )

                self.optimizer.step()

            # Calculate metrics: confusion matrix elements
            _, preds = torch.max(outputs, dim=1)
            targets = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            
            TP, FP, TN, FN = get_confusion_elements(targets, preds, n_classes=self.config.data.output_classes)
            TP_sum += TP
            FP_sum += FP
            TN_sum += TN
            FN_sum += FN

            # Update Metrics
            loss_meter.update(loss.item(), inputs.size(0))
            batch_time.update(time.time() - start)
            if torch.cuda.is_available():
                cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))

            # Log batch-level metrics only if main process
            if is_main_process() and self.writer and batch_idx % 10 == 0:
                global_step = epoch * total_batches + batch_idx
                self.writer.add_scalar("Loss/train_batch", loss.item(), global_step)

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("Learning_Rate", current_lr, global_step)

                if self.use_amp and self.scaler is not None:
                    self.writer.add_scalar('AMP/loss_scale', self.scaler.get_scale(), global_step)

            # Update progress bar
            if is_main_process():
                monitor = {
                    'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})',
                    'CUDA': f'{cuda_mem.val:.2f} ({cuda_mem.avg:.2f})'
                }
                if self.use_amp and self.scaler is not None:
                    monitor['Scale'] = f'{self.scaler.get_scale():.0f}'
                pbar.set_postfix(monitor)

            start = time.time()

        # Log epoch-level metrics
        if self.writer:
            TP_sum = TP_sum.sum()
            FP_sum = FP_sum.sum()
            TN_sum = TN_sum.sum()
            FN_sum = FN_sum.sum()
            precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0
            recall = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0
            accuracy = (TP_sum + TN_sum) / (TP_sum + FP_sum + TN_sum + FN_sum) if (TP_sum + FP_sum + TN_sum + FN_sum) > 0 else 0

            self.writer.add_scalar("Loss/train", loss_meter.avg, epoch)
            self.writer.add_scalar("Metrics/precision", precision, epoch)
            self.writer.add_scalar("Metrics/recall", recall, epoch)
            self.writer.add_scalar("Metrics/accuracy", accuracy, epoch)
            self.writer.add_scalar('Memory/avg_cuda_memory_gb', cuda_mem.avg, epoch)

        # Gather all metrics across processes
        if is_distributed():
            process_metrics = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                batch_time.avg,
                data_time.avg,
                loss_meter.sum,
                loss_meter.count,
                TP_sum,
                FN_sum,
                FP_sum,
                TN_sum,
            ], device=self.device)

            gathered_metrics = gather_tensor(process_metrics)

            if is_main_process():
                world_size = get_world_size()
                all_avg_losses = gathered_metrics[0::10]
                all_cuda_mem = gathered_metrics[1::10]
                all_batch_times = gathered_metrics[2::10]
                all_data_times = gathered_metrics[3::10]
                all_total_losses = gathered_metrics[4::10]
                all_sample_counts = gathered_metrics[5::10]
                all_TPs = gathered_metrics[6::10]
                all_FNs = gathered_metrics[7::10]
                all_FPs = gathered_metrics[8::10]
                all_TNs = gathered_metrics[9::10]

                # Compute global averages 
                total_samples = all_sample_counts.sum().item()
                global_avg_loss = all_total_losses.sum().item() / total_samples if total_samples > 0 else 0

                global_precision = all_TPs.sum().item() / (
                    all_TPs.sum().item() + all_FPs.sum().item()) if (all_TPs.sum().item() + all_FPs.sum().item()) > 0 else 0
                global_recall = all_TPs.sum().item() / (
                    all_TPs.sum().item() + all_FNs.sum().item()) if (all_TPs.sum().item() + all_FNs.sum().item()) > 0 else 0
                global_accuracy = (all_TPs.sum().item() + all_TNs.sum().item()) / (
                    all_TPs.sum().item() + all_FPs.sum().item() +
                    all_TNs.sum().item() + all_FNs.sum().item()
                ) if (all_TPs.sum().item() + all_FPs.sum().item() +
                    all_TNs.sum().item() + all_FNs.sum().item()) > 0 else 0
                
                # Log global metrics
                self.writer.add_scalar("Loss/train_global", global_avg_loss, epoch)
                self.writer.add_scalar('Loss/train_avg_across_ranks', all_avg_losses.mean().item(), epoch)
                self.writer.add_scalar('Loss/train_min', all_avg_losses.min().item(), epoch)
                self.writer.add_scalar('Loss/train_max', all_avg_losses.max().item(), epoch)
                self.writer.add_scalar('Loss/train_std', all_avg_losses.std().item(), epoch)

                self.writer.add_scalar('Memory/cuda_memory_avg', all_cuda_mem.mean().item(), epoch)
                self.writer.add_scalar('Memory/cuda_memory_max', all_cuda_mem.max().item(), epoch)

                self.writer.add_scalar('Timing/batch_time_avg', all_batch_times.mean().item(), epoch)
                self.writer.add_scalar('Timing/data_time_avg', all_data_times.mean().item(), epoch)

                self.writer.add_scalar("Metrics/precision_global", global_precision, epoch)
                self.writer.add_scalar("Metrics/recall_global", global_recall, epoch)
                self.writer.add_scalar("Metrics/accuracy_global", global_accuracy, epoch)

                try:
                    # Log histograms for better visualization
                    self.writer.add_histogram('Distributions/loss_per_rank', all_avg_losses.cpu(), epoch)
                    self.writer.add_histogram('Distributions/memory_per_rank', all_cuda_mem.cpu(), epoch)
                except Exception as e: 
                    print(e)
                return {'train_loss': global_avg_loss}
            else:
                return {'train_loss': loss_meter.avg}
            
        else:
            # Single process training
            if is_main_process() and self.writer:
                self.writer.add_scalar('Loss/train', loss_meter.avg, epoch)
                self.writer.add_scalar('Memory/cuda_memory_avg', cuda_mem.avg, epoch)
            return {'train_loss': loss_meter.avg}

    def _validate_epoch(self, epoch):
        self.model.eval()

        loss_meter = AverageMeter()
        cuda_mem = AverageMeter()
        TP_sum = 0
        FP_sum = 0
        TN_sum = 0
        FN_sum = 0

        with torch.no_grad():
            if is_main_process():
                pbar = tqdm(self.val_loader)
                pbar.set_description(f'[VALIDATION] Epoch {epoch + 1}')
            else:
                pbar = self.val_loader

            for batch_idx, batch in enumerate(pbar):
                inputs = batch['time_proj'].to(self.device)
                labels = batch['class'].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Calculate metrics: confusion matrix elements
                _, preds = torch.max(outputs, dim=1)
                targets = labels.cpu().numpy()
                preds = preds.cpu().numpy()

                TP, FP, TN, FN = get_confusion_elements(targets, preds, n_classes=self.config.data.output_classes)
                TP_sum += TP
                FP_sum += FP
                TN_sum += TN
                FN_sum += FN

                loss_meter.update(loss.item(), inputs.size(0))
                
                if torch.cuda.is_available():
                    cuda_mem.update(torch.cuda.max_memory_allocated(device=None) / (1024 * 1024 * 1024))

                if is_main_process() and self.writer and batch_idx % 10 == 0:
                    global_step = epoch * len(self.val_loader) + batch_idx
                    self.writer.add_scalar("Loss/val_batch", loss.item(), global_step)

                if is_main_process():
                    pbar.set_postfix({'Loss': f'{loss_meter.val:.4f} ({loss_meter.avg:.4f})'})

        # Log epoch-level metrics
        if is_main_process() and self.writer:
            TP_sum = TP_sum.sum()
            FP_sum = FP_sum.sum()
            TN_sum = TN_sum.sum()
            FN_sum = FN_sum.sum()
            precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0
            recall = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0
            accuracy = (TP_sum + TN_sum) / (TP_sum + FP_sum + TN_sum + FN_sum) if (TP_sum + FP_sum + TN_sum + FN_sum) > 0 else 0

            self.writer.add_scalar("Loss/val", loss_meter.avg, epoch)
            self.writer.add_scalar("Metrics/precision", precision, epoch)
            self.writer.add_scalar("Metrics/recall", recall, epoch)
            self.writer.add_scalar("Metrics/accuracy", accuracy, epoch)
            self.writer.add_scalar('Memory/cuda_memory_avg', cuda_mem.avg, epoch)

        # Gather all metrics across processes
        if is_distributed():
            process_metrics = torch.tensor([
                loss_meter.avg,
                cuda_mem.avg,
                TP_sum,
                FN_sum,
                FP_sum,
                TN_sum,
            ], device=self.device)

            gathered_metrics = gather_tensor(process_metrics)

            if is_main_process():
                world_size = get_world_size()
                all_avg_losses = gathered_metrics[0::6]
                all_cuda_mem = gathered_metrics[1::6]
                all_TPs = gathered_metrics[2::6]
                all_FNs = gathered_metrics[3::6]
                all_FPs = gathered_metrics[4::6]
                all_TNs = gathered_metrics[5::6]

                # Compute global averages 
                total_samples = len(self.val_loader.dataset)
                global_avg_loss = all_avg_losses.sum().item() / total_samples if total_samples > 0 else 0

                global_precision = all_TPs.sum().item() / (
                    all_TPs.sum().item() + all_FPs.sum().item()) if (all_TPs.sum().item() + all_FPs.sum().item()) > 0 else 0
                global_recall = all_TPs.sum().item() / (
                    all_TPs.sum().item() + all_FNs.sum().item()) if (all_TPs.sum().item() + all_FNs.sum().item()) > 0 else 0
                global_accuracy = (all_TPs.sum().item() + all_TNs.sum().item()) / (
                    all_TPs.sum().item() + all_FPs.sum().item() +
                    all_TNs.sum().item() + all_FNs.sum().item()
                ) if (all_TPs.sum().item() + all_FPs.sum().item() +
                    all_TNs.sum().item() + all_FNs.sum().item()) > 0 else 0
                
                # Log global metrics
                self.writer.add_scalar("Loss/val_global", global_avg_loss, epoch)
                self.writer.add_scalar('Loss/val_avg_across_ranks', all_avg_losses.mean().item(), epoch)
                self.writer.add_scalar('Loss/val_min', all_avg_losses.min().item(), epoch)
                self.writer.add_scalar('Loss/val_max', all_avg_losses.max().item(), epoch)
                self.writer.add_scalar('Loss/val_std', all_avg_losses.std().item(), epoch)

                self.writer.add_scalar('Memory/cuda_memory_avg', all_cuda_mem.mean().item(), epoch)
                self.writer.add_scalar('Memory/cuda_memory_max', all_cuda_mem.max().item(), epoch)
                self.writer.add_scalar("Metrics/precision_global", global_precision, epoch)
                self.writer.add_scalar("Metrics/recall_global", global_recall, epoch)
                self.writer.add_scalar("Metrics/accuracy_global", global_accuracy, epoch)

                return {
                    'val_loss': global_avg_loss,
                    'precision': global_precision,
                    'recall': global_recall,
                    'accuracy': global_accuracy
                }
            else:
                return {
                    'val_loss': loss_meter.avg,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy
                }
            
    def _load_ckpt(self, checkpoint_path):
        pass

    def save_run_config(self, config):

        if not is_main_process():
            return
        try:
            config_dict = self._config_to_dict(config)
            config_path = self.output_dir / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

                print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Failed to save configuration: {e}")

    def _config_to_dict(self, obj):
        if hasattr(obj, '__dict__'):
            # If it's an object with attributes, convert to dict
            result = {}
            for key, value in vars(obj).items():
                # Skip private attributes (starting with _)
                if not key.startswith('_'):
                    result[key] = self._config_to_dict(value)
            return result
        elif isinstance(obj, (list, tuple)):
            # If it's a list or tuple, convert each element
            return [self._config_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            # If it's already a dict, convert its values
            return {key: self._config_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            # Primitive types can be used as-is
            return obj
        else:
            # For any other types, convert to string
            return str(obj)
        
    def _create_optimizer(self):
        opt_config = self.config.training.optimizer
        effective_batch_size = self.config.training.batch_size * self.world_size
        lr_scale = effective_batch_size / self.config.training.batch_size

        if opt_config.name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.learning_rate * lr_scale,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config.learning_rate * lr_scale,
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config.name}")
        
        return optimizer
    
    def _create_scheduler(self):
        sched_config = self.config.training.scheduler

        if sched_config.name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=0
            )
        elif sched_config.name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            return None