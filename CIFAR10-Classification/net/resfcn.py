import numbers
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResFCN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.conv_block1 = BasicBlock(3, 64)
        self.downsample1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_block2 = BasicBlock(64, 128)
        self.downsample2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_block3 = BasicBlock(128, 256)
        self.downsample3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_block4 = BasicBlock(256, 512)
        self.downsample4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_block5 = BasicBlock(512, 512)
        self.downsample5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
            
        # 采用5倍下采样将32x32的图片采样为1x1的特征
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, self.n_classes, 3, 1, 1),
        )

    def forward(self, x):
        output = self.downsample1(self.conv_block1(x))
        output = self.downsample2(self.conv_block2(output))
        output = self.downsample3(self.conv_block3(output))
        output = self.downsample4(self.conv_block4(output))
        output = self.downsample5(self.conv_block5(output))
        output = self.classifier(output)
        output = output.reshape(-1, self.n_classes)

        return output


 