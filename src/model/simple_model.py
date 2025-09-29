import torch
import torch.nn as nn
import torchvision.models as models

    
class SimpleCNN(nn.Module):
    # 4 CNN 3x3, 2^n 
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        out_1 = in_channels * 4
        out_2 = out_1 * 2
        out_3 = out_2 * 2
        out_4 = out_3 * 2

        self.model = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_1, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_1),
            nn.ReLU(),

            nn.Conv2d(out_1, out_2, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_2),
            nn.ReLU(),

            nn.Conv2d(out_2, out_3, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_3),
            nn.ReLU(),

            nn.Conv2d(out_3, out_4, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_4),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_4, out_3),
            nn.Dropout(0.5),
            nn.Linear(out_3, out_2),
            nn.Linear(out_2, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x    
    
class MidCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MidCNN, self).__init__()
        out_1 = in_channels * 4
        out_2 = out_1 * 4
        out_3 = out_2 * 2
        out_4 = out_3 * 2

        self.model = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_1, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_1),
            nn.ReLU(),

            nn.Conv2d(out_1, out_2, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_2),
            nn.ReLU(),

            nn.Conv2d(out_2, out_3, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_3),
            nn.ReLU(),

            nn.Conv2d(out_3, out_4, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_4),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_4, out_3),
            nn.Dropout(0.5),
            nn.Linear(out_3, out_2),
            nn.Linear(out_2, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x 

if __name__ == '__main__':
    # model = RadarEdgeNetwork(in_channels=2, filters1=64, filters2=32)
    model = SimpleCNN(in_channels=2, num_classes=12)
    input_tensor = torch.randn(8, 2, 32, 100)
    output = model(input_tensor)
    print(output.shape)
    print(output)
