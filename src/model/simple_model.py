import torch
import torch.nn as nn

class DepthwiseExpansionModule(nn.Module):
    def __init__(self, in_channels, alpha=1):
        super(DepthwiseExpansionModule, self).__init__()

        self.depthwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        num_filter_1 = 2 * alpha * in_channels
        self.pointwise1 = nn.Conv2d(in_channels, num_filter_1, kernel_size=1, stride=1)
        num_filter_2 = 2 * alpha * num_filter_1
        self.pointwise2 = nn.Conv2d(num_filter_1, num_filter_2, kernel_size=1, stride=1)
        
        self.depthwise2 = nn.Conv2d(num_filter_2, num_filter_2, kernel_size=3, stride=1, padding=1, groups=num_filter_2)
        num_filter_3 = 2 * alpha * num_filter_2
        self.final_pointwise = nn.Conv2d(num_filter_2, num_filter_3, kernel_size=1, stride=2)

        self.activation = nn.ReLU6()

    def forward(self, x):
        x = self.activation(self.depthwise1(x))
        x = self.activation(self.pointwise1(x))
        x = self.activation(self.pointwise2(x))
        x = self.activation(self.depthwise2(x))
        x = self.activation(self.final_pointwise(x))
        return x

class RadarEdgeNetwork(nn.Module):
    def __init__(self, in_channels=3, filters1=64, filters2=32, alpha=1, num_classes = 11):
        super(RadarEdgeNetwork, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels, filters1, kernel_size=3, stride=1, padding= 1)  #??
        self.depthwise_expansion1 = DepthwiseExpansionModule(filters1, alpha=alpha)
        out_channel_exp1 = (2*alpha)**3 * filters1

        self.conv_layer_2 = nn.Conv2d(out_channel_exp1, filters2, kernel_size=3, stride=1, padding= 1)  #??
        self.depthwise_expansion2 = DepthwiseExpansionModule(filters2, alpha=alpha)
        out_channel_exp2 = (2*alpha)**3 * filters1
        self.maxpool2d = nn.MaxPool2d(kernel_size=3)   # ??? unknown settings
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_channel_exp2*100, num_classes)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.depthwise_expansion1(x)
        x = self.conv_layer_2(x)
        x = self.depthwise_expansion2(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.fc(x)

        # print(f"x output_shape: {x.shape}")

        return x
    
class SimpleCNN(nn.Module):
    # 4 CNN 3x3, 2^n 
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        out_1 = in_channels*4
        self.cnn_1 = nn.Conv2d(in_channels, out_1, kernel_size=3, stride=1)
        out_2 = out_1*2
        self.cnn_2 = nn.Conv2d(out_1, out_2, kernel_size=3, stride=1)
        out_3 = out_2*2
        self.cnn_3 = nn.Conv2d(out_2, out_3, kernel_size=3, stride=1)
        out_4 = out_3*2
        self.cnn_4 = nn.Conv2d(out_3, out_4, kernel_size=3, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(out_4, out_3)
        self.fc2 = nn.Linear(out_3, num_classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.cnn_1(x)
        x = self.activation(x)
        x = self.cnn_2(x)
        x = self.activation(x)
        x = self.cnn_3(x)
        x = self.activation(x)
        x = self.cnn_4(x)
        x = self.activation(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x    

if __name__ == '__main__':
    # model = RadarEdgeNetwork(in_channels=2, filters1=64, filters2=32)
    model = SimpleCNN(in_channels=2, num_classes=12)
    input_tensor = torch.randn(8, 2, 32, 100)
    output = model(input_tensor)
    print(output.shape)
    print(output)
