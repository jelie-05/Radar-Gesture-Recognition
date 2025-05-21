from model.simple_model import RadarEdgeNetwork, SimpleCNN
import torch
import torch.nn as nn
from utils.dataloader_rdatm import SoliDataset, DataGenerator
from torch.utils.data import random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from sklearn.metrics import precision_score, recall_score, accuracy_score

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

def train_model():
    dataset = SoliDataset(data_path='data/SoliData/dsp', resolution=(32, 32), num_channels=3)
    train_size = int(0.6*len(dataset))
    val_size = int(0.4*len(dataset))
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_size, val_size])

    dataloader_train = DataGenerator(train_dataset, batch_size=32, shuffle=True, max_length=75, num_workers=4, drop_last=True).get_loader()
    dataloader_val = DataGenerator(val_dataset, batch_size=32, shuffle=True, max_length=75, num_workers=4, drop_last=True).get_loader()
    
    writer = SummaryWriter(log_dir='runs')

    # model = RadarEdgeNetwork(in_channels=2, filters1=64, filters2=32, alpha=1, num_classes=12)
    model = SimpleCNN(in_channels=2, num_classes=12)
    print(summary(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    best_loss = 99.9

    for epoch in range(num_epochs):
        train_loop = create_tqdm_bar(dataloader_train, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]')

        training_loss = 0
        validation_loss = 0
        precision_train = 0
        recall_train = 0
        accuracy_train = 0
        precision_val = 0
        recall_val = 0
        accuracy_val = 0
        

        for train_iter, batch in train_loop:
            batch_videos = batch['rdtm']
            batch_classes = batch['class']
            batch_videos = batch_videos.to(device)
            batch_classes = batch_classes.to(device).squeeze(1)
            # print(f"target: {batch_classes}")

            optimizer.zero_grad()
            outputs = model(batch_videos)

            loss = loss_fn(outputs, batch_classes)
            loss.backward()
            optimizer.step()

            v, idx = torch.max(outputs, dim=1)

            target = batch_classes.cpu().numpy()
            pred = idx.cpu().numpy()
            
            precision_train += precision_score(target, pred, average='macro', zero_division=0)
            recall_train += recall_score(target, pred, average='macro', zero_division=0)
            accuracy_train += accuracy_score(target, pred)

            v, idx = torch.max(outputs, dim=1)

            training_loss += loss.item()

        train_loss_epoch = training_loss/len(dataloader_train)
        writer.add_scalar("Loss/train", train_loss_epoch, epoch)

        train_precision_epoch = precision_train/len(dataloader_train)
        train_recall_epoch = recall_train/len(dataloader_train)
        train_accuracy_epoch = accuracy_train/len(dataloader_train)
        writer.add_scalar("Metrics/precision", train_precision_epoch, epoch)
        writer.add_scalar("Metrics/recall", train_recall_epoch, epoch)
        writer.add_scalar("Metrics/accuracy", train_accuracy_epoch, epoch)

        val_loop = create_tqdm_bar(dataloader_val, desc=f'Validation Epoch [{epoch + 1}/{num_epochs}]')

        with torch.no_grad():
            for val_iter, batch in val_loop:
                batch_videos = batch['rdtm']
                batch_classes = batch['class']
                batch_videos = batch_videos.to(device)
                batch_classes = batch_classes.to(device).squeeze(1)

                outputs = model(batch_videos)

                loss_val = loss_fn(outputs, batch_classes)
                validation_loss += loss_val.item()

                v, idx = torch.max(outputs, dim=1)
                # print(f"idx: {idx}")

                target = batch_classes.cpu().numpy()
                pred = idx.cpu().numpy()
                
                precision_val += precision_score(target, pred, average='macro', zero_division=0)
                recall_val += recall_score(target, pred, average='macro', zero_division=0)
                accuracy_val += accuracy_score(target, pred)

            val_loss_epoch = validation_loss / len(dataloader_val)
            writer.add_scalar("Loss/val", val_loss_epoch, epoch)

            val_precision_epoch = precision_train/len(dataloader_train)
            val_recall_epoch = recall_train/len(dataloader_train)
            val_accuracy_epoch = accuracy_train/len(dataloader_train)
            writer.add_scalar("Metrics/precision_val", val_precision_epoch, epoch)
            writer.add_scalar("Metrics/recall_val", val_recall_epoch, epoch)
            writer.add_scalar("Metrics/accuracy_val", val_accuracy_epoch, epoch)
            
            if val_loss_epoch < best_loss:
                best_loss = val_loss_epoch
                torch.save(model.state_dict(), 'runs/trained_models/radar_edge_network-current_best.pth')

    print("Training complete.")
    torch.save(model.state_dict(), 'runs/trained_models/radar_edge_network.pth')
if __name__ == "__main__":
    train_model()
