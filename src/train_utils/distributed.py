"""Distributed training utilities."""
import os
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional, Union


def is_distributed() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process."""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize the process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set GPU device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        
        return device, rank, world_size, local_rank
    else:
        # Single GPU / CPU training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training environment."""
    if is_distributed():
        dist.destroy_process_group()


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """
    Reduce a dictionary of tensors across all processes.
    
    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average the values across processes
        
    Returns:
        Dictionary with reduced tensors
    """
    if not is_distributed():
        return input_dict
    
    world_size = get_world_size()
    
    # Convert all values to tensors if they aren't already
    tensor_dict = {}
    for key, value in input_dict.items():
        if not isinstance(value, torch.Tensor):
            tensor_dict[key] = torch.tensor(value, dtype=torch.float32)
        else:
            tensor_dict[key] = value.clone()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            tensor_dict[key] = tensor_dict[key].cuda()
    
    # Reduce all tensors
    reduced_dict = {}
    for key, tensor in tensor_dict.items():
        dist.all_reduce(tensor)
        if average:
            tensor /= world_size
        reduced_dict[key] = tensor.item()
    
    return reduced_dict


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        average: Whether to average the tensor across processes
        
    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor
    
    world_size = get_world_size()
    
    # Clone tensor to avoid in-place modifications
    rt = tensor.clone()
    
    # Move to GPU if available and not already there
    if torch.cuda.is_available() and not rt.is_cuda:
        rt = rt.cuda()
    
    # Reduce across all processes
    dist.all_reduce(rt)
    
    if average:
        rt /= world_size
    
    return rt


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Concatenated tensor from all processes (only valid on rank 0)
    """
    if not is_distributed():
        return tensor
    
    world_size = get_world_size()
    
    # Move to GPU if available and not already there
    if torch.cuda.is_available() and not tensor.is_cuda:
        tensor = tensor.cuda()
    
    # Create list to store gathered tensors
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather all tensors
    dist.all_gather(gathered_tensors, tensor)
    
    # Concatenate tensors (this will be the same on all ranks)
    return torch.cat(gathered_tensors, dim=0)


def synchronize():
    """Synchronize all processes."""
    if not is_distributed():
        return
    dist.barrier()


def save_on_master(obj: Any, filename: str):
    """Save object only on the master process."""
    if is_main_process():
        torch.save(obj, filename)


def init_seeds(seed: int, rank: int):
    """Initialize random seeds for reproducible training."""
    import random
    import numpy as np
    
    # Set base seed
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False