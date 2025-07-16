from src.model.simple_model import RadarEdgeNetwork, SimpleCNN
import torch
import torch.nn as nn
from src.train_utils.dataset import IFXRadarDataset, IFXDataGen
from ifxAvian import Avian
from torch.utils.data import random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from sklearn.metrics import precision_score, recall_score, accuracy_score
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savename', type=str, required=True, help='Location of training data')

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

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

def train_model(datadir, num_classes, in_channels, save_name, epochs, observation_length):
    # Device configuration
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
    config = {'dev_config': dev_config, 
              'num_rx_antennas': 3, 
              'num_beams': 32,
              'max_angle_degrees': 40}
    
    # Create dataset
    dataset = IFXRadarDataset(config, root_dir='/home/swadiryus/projects/dataset/radar_gesture_dataset')
    generator1 = torch.Generator().manual_seed(1)

    total_samples = len(dataset)  # e.g., 2123
    train_ratio = 0.6
    train_size = int(train_ratio * total_samples)       # e.g., 1273
    val_size = total_samples - train_size 

    print(f"Train size: {train_size}, Validation size: {val_size}")
    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[train_size, val_size], generator=generator1)

    observation_length = observation_length

    dataloader_train = IFXDataGen(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True).get_loader()
    dataloader_val = IFXDataGen(val_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True).get_loader()
    
    writer = SummaryWriter(log_dir=f'runs/run_{save_name}')

    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    print(summary(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    best_loss = 99.9

    for epoch in range(num_epochs):
        train_loop = create_tqdm_bar(dataloader_train, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]')

        training_loss = 0
        validation_loss = 0

        TP_sum = 0
        FP_sum = 0
        TN_sum = 0
        FN_sum = 0
        

        for train_iter, batch in train_loop:
            batch_videos = batch['rdtm']
            batch_classes = batch['class']
            batch_videos = batch_videos.to(device)
            batch_classes = batch_classes.to(device)

            optimizer.zero_grad()
            outputs = model(batch_videos)

            loss = loss_fn(outputs, batch_classes)
            loss.backward()
            optimizer.step()

            v, idx = torch.max(outputs, dim=1)

            target = batch_classes.cpu().numpy()
            pred = idx.cpu().numpy()

            TP, FP, TN, FN = get_confusion_elements(target, pred, n_classes=12)
            TP_sum += TP
            FP_sum += FP
            TN_sum += TN
            FN_sum += FN

            training_loss += loss.item()

        TP_sum = TP_sum.sum()
        FP_sum = FP_sum.sum()
        FN_sum = FN_sum.sum()
        TN_sum = TN_sum.sum()

        train_loss_epoch = training_loss/len(dataloader_train)
        writer.add_scalar("Loss/train", train_loss_epoch, epoch)

        train_precision_epoch = (TP_sum)/(TP_sum+FP_sum)
        train_recall_epoch = TP_sum/(TP_sum+FN_sum)
        train_accuracy_epoch = (TP_sum+TN_sum)/(TP_sum+FP_sum+TN_sum+FN_sum)
        writer.add_scalar("Metrics/precision", train_precision_epoch, epoch)
        writer.add_scalar("Metrics/recall", train_recall_epoch, epoch)
        writer.add_scalar("Metrics/accuracy", train_accuracy_epoch, epoch)

        TP_sum = 0
        FP_sum = 0
        TN_sum = 0
        FN_sum = 0

        val_loop = create_tqdm_bar(dataloader_val, desc=f'Validation Epoch [{epoch + 1}/{num_epochs}]')

        with torch.no_grad():
            for val_iter, batch in val_loop:
                batch_videos = batch['rdtm']
                batch_classes = batch['class']
                batch_videos = batch_videos.to(device)
                batch_classes = batch_classes.to(device)
                outputs = model(batch_videos)

                loss_val = loss_fn(outputs, batch_classes)
                validation_loss += loss_val.item()

                v, idx = torch.max(outputs, dim=1)
                # print(f"idx: {idx}")

                target = batch_classes.cpu().numpy()
                pred = idx.cpu().numpy()
                
                # precision_val += precision_score(target, pred, average='macro', zero_division=0)
                # recall_val += recall_score(target, pred, average='macro', zero_division=0)
                # accuracy_val += accuracy_score(target, pred)

                TP, FP, TN, FN = get_confusion_elements(target, pred, n_classes=12)
                TP_sum += TP
                FP_sum += FP
                TN_sum += TN
                FN_sum += FN

            val_loss_epoch = validation_loss / len(dataloader_val)
            writer.add_scalar("Loss/val", val_loss_epoch, epoch)

            # val_precision_epoch = precision_val/len(dataloader_train)
            # val_recall_epoch = recall_val/len(dataloader_train)
            # val_accuracy_epoch = accuracy_val/len(dataloader_train)
            TP_sum = TP_sum.sum()
            FP_sum = FP_sum.sum()
            FN_sum = FN_sum.sum()
            TN_sum = TN_sum.sum()

            val_precision_epoch = (TP_sum)/(TP_sum+FP_sum)
            val_recall_epoch = TP_sum/(TP_sum+FN_sum)
            val_accuracy_epoch = (TP_sum+TN_sum)/(TP_sum+FP_sum+TN_sum+FN_sum)

            writer.add_scalar("Metrics/precision_val", val_precision_epoch, epoch)
            writer.add_scalar("Metrics/recall_val", val_recall_epoch, epoch)
            writer.add_scalar("Metrics/accuracy_val", val_accuracy_epoch, epoch)
            
            if val_loss_epoch < best_loss:
                best_loss = val_loss_epoch
                torch.save(model.state_dict(), f'runs/trained_models/{save_name}-current_best.pth')

    with open(f'runs/trained_models/{save_name}-idx_mapping.pkl', 'wb') as f:
        pickle.dump(dataset.get_mapping(), f)


    print("Training complete.")
    torch.save(model.state_dict(), f'runs/trained_models/{save_name}-last.pth')
    
if __name__ == "__main__":
    args = parser.parse_args()

    datadir = 'data/recording'
    in_channels = 3
    num_classes = 5
    epochs = 50
    observation_length = 10

    train_model(datadir=datadir, in_channels=in_channels, num_classes=num_classes, epochs=epochs, save_name=args.savename, observation_length=observation_length)
