from src.train_utils.dataset import IFXRadarDataset
from torch.utils.data import random_split, DataLoader, DistributedSampler
import torch

def split_dataset(dataset, split=(0.6,0.2,0.2), seed=42):
    """ Splits the dataset into training, validation, and test sets. """

    total_samples = len(dataset)
    train_size = int(total_samples * split[0])
    val_size = int(total_samples * split[1])
    test_size = total_samples - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size], 
                        generator=torch.Generator().manual_seed(seed))

def custom_collate_fn(self, batch):
    inputs, labels = zip(*batch)
    inputs_tensor = torch.stack(inputs, dim=0)  # Shape: (batch_size, 3, H, W)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    batch = {
        'inputs': inputs_tensor,
        'labels': labels_tensor
    }
    return batch

def get_dataloaders(
        config, 
        radar_config,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        ):
    dataset = IFXRadarDataset(radar_config, root_dir=config.data.dataset_path)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, seed=config.training.seed)

    # Samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=config.data.drop_last,
        collate_fn=custom_collate_fn
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            prefetch_factor=config.data.prefetch_factor,
            drop_last=config.data.drop_last,
            collate_fn=custom_collate_fn
        )
    else:
        val_loader = None

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            persistent_workers=config.data.persistent_workers,
            prefetch_factor=config.data.prefetch_factor,
            drop_last=config.data.drop_last,
            collate_fn=custom_collate_fn
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
    

