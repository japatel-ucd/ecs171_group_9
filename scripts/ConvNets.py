import torch
import torch.nn as nn
import torch.nn.functional as F


class Alex(nn.Module): 
    def __init__(self):
        super(Alex, self).__init__()
        # input shape: (B, 3, 64, 64)
        #     B: batch_size
        #     3: RGB channels
        #     64,64: fixed image size (64, 64)

        # convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(True),  # (B,  1, 64,64) → (B, 16, 64,64)
            nn.MaxPool2d(kernel_size=2), nn.Dropout(0.25),              # (B, 16, 64,64) → (B, 16, 32,32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(True), # (B, 16, 32,32) → (B, 32, 32,32)
            nn.MaxPool2d(kernel_size=2), nn.Dropout(0.25),              # (B, 32, 32,32) → (B, 32, 16,16)
        )        
        # fully-connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),              # (B, 32, 16,16) → (B, 32×16×16)
            nn.Linear(32*16*16, 256),  # (B, 32×16×16) → (B,256)
            nn.ReLU(True), nn.Dropout(0.5)
        )
        # last layer for classification
        self.final = nn.Linear(256, 1)      # (B, 256) → (B, 1)
        # activation function of last layer
        self.activation = nn.Sigmoid() # (B, 1) → (B, 1)

    def forward(self, x):
        # x: input tensor, an image batch of size B×C×H×W
        x = self.conv(x)
        x = self.fc(x)
        x = self.final(x)
        x = self.activation(x)
        return x


# inspired by GoogLeNet
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2: ])


class Inception(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, channel_3, channel_4):
        super(Inception, self).__init__()
        # input:
        #     in_channel: integer, no. of input channel
        #     channel_1: integer, no. of 1×1 kernals
        #     channel_2: integer tuple, (no. of 1×1 kernals, no. of 3×3 kernals)
        #     channel_3: integer tuple, (no. of 1×1 kernals, no. of 5×5 kernals)
        #     channel_4: integer, no. of 1×1 kernals

        # convolutional layers, which do not alter input size
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channel_1, kernel_size=1), nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channel, channel_2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(channel_2[0], channel_2[1], kernel_size=3, padding=1), nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channel, channel_3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(channel_3[0], channel_3[1], kernel_size=5, padding=2), nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, channel_4, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)
        x_3 = self.conv_3(x)
        x_4 = self.conv_4(x)
        return torch.cat((x_1,x_2,x_3,x_4), dim=1)


class GoogLe(nn.Module):
    def __init__(self):
        super(GoogLe, self).__init__()
        # input shape: (B, 3, 224, 224)
        #     B: batch_size
        #     3: RGB channels
        #     224,224: fixed image size

        self.inception = nn.Sequential(
            Inception(3, 8, (1,8), (1,8), 8),                  # (B,  3, 224,224) → (B, 32, 224,224)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (B, 32, 224,224) → (B, 32, 112,112)
            Inception(32, 16, (16,16), (16,16), 16),           # (B, 32, 112,112) → (B, 64, 112,112)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # (B, 64, 112,112) → (B, 64,  56,56 )
            Inception(64, 24, (16,24), (16,24), 24),           # (B, 64,  56,56 ) → (B, 96,  56,56 )
            GlobalAvgPool2d()                                  # (B, 96,  56,56 ) → (B, 96,   1,1  )
        )

        self.final = nn.Linear(96, 1)  # (B, 96) → (B, 1)

        self.activation = nn.Sigmoid() # (B, 1 ) → (B, 1)

    def forward(self, x):
        x = self.inception(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        x = self.activation(x)
        return x


# inspired by DenseNet
class BN_Conv2d(nn.Module):
    """ BN_Conv_ReLU """
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object,
                 stride: object, padding: object, dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))


class DenseBlock(nn.Module):
    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = nn.ModuleList()
        for i in range(self.num_layers):
            layer_list.append(nn.Sequential(
                BN_Conv2d(self.k0+i*self.k, 4*self.k, 1, 1, 0),
                BN_Conv2d(4 * self.k, self.k, 3, 1, 1)
            ))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature, out), 1)
        return out


class Dense(nn.Module):
    def __init__(self, layers: object, k, theta, num_classes) -> object:
        super(Dense, self).__init__()
        # parameters
        self.layers = layers
        self.k = k
        self.theta = theta
        # layers
        self.conv = BN_Conv2d(3, 2*k, 7, 2, 3)
        self.blocks, patches = self.__make_blocks(2*k)
        self.final = nn.Linear(patches, num_classes)
        self.activation = nn.Sigmoid()

    def __make_transition(self, in_chls):
        out_chls = int(self.theta*in_chls)
        return nn.Sequential(
            BN_Conv2d(in_chls, out_chls, 1, 1, 0),
            nn.AvgPool2d(2)
        ), out_chls

    def __make_blocks(self, k0):
        layers_list = nn.ModuleList()
        patches = 0
        for i in range(len(self.layers)):
            layers_list.append(DenseBlock(k0, self.layers[i], self.k))
            patches = k0+self.layers[i]*self.k # output feature patches from DenseBlock
            if i != len(self.layers)-1:
                transition, k0 = self.__make_transition(patches)
                layers_list.append(transition)
        return nn.Sequential(*layers_list), patches

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, 3, 2, 1)
        #print(out.shape)
        x = self.blocks(x)
        #print(out.shape)
        x = F.avg_pool2d(x, 7)
        #print(out.shape)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        x = self.activation(x)
        return x