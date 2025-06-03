from utils.dataloader_rdatm import NumpyDataset, SoliDataset, DataGenerator
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# soli
dataset = SoliDataset(data_path='data/SoliData/dsp', resolution=(32, 32), num_channels=3)
dataloader =  DataGenerator(dataset, batch_size=8, shuffle=True, max_length=30, num_workers=4, drop_last=True).get_loader()
for batch_data in dataloader:
    print(batch_data['rdtm'].shape)   # torch.Size([8, 30, H, W])
    print(batch_data['class'].shape)       # tensor of shape [8]
    break

dataset = NumpyDataset()
dataloader = DataGenerator(dataset, batch_size=8, shuffle=True, max_length=15, num_workers=4, drop_last=True).get_loader()
for batch_data in dataloader:
    print(batch_data['rdtm'].shape)   # torch.Size([8, 2, 30, H, W])
    print(batch_data['class'].shape)       # tensor of shape [8]
    label = batch_data['class']


    for i in range(len(label)):
        print(f"Label {i}: {label[i].item()}")
        print(f"Class name {i}: {dataset.get_class_name(label[i].item())}")
        label_cur = dataset.get_class_name(label[i].item())
        vmin = 0.0
        vmax = 0.8
        plt.figure(figsize=(18, 10))
        plt.imshow(batch_data['rdtm'][i][0], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f"rtm {label_cur}")
        plt.figure(figsize=(18, 10))
        plt.imshow(batch_data['rdtm'][i][1], aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f"dtm{label_cur}")
    
    plt.show()
    # print("Numeric label:", label)
    # print("Class name:", dataset.get_class_name(2))
    break


