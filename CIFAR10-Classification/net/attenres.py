import torch.nn as nn
import torch
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=padding)
        self.cbr = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=inplace))
    
    def forward(self,x):
        return self.cbr(x)

class CR(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=padding)
        self.cr = nn.Sequential(conv, nn.ReLU(inplace=inplace))
    
    def forward(self,x):
        return self.cr(x)

class CB(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels,out_channels,k_size,stride,padding)
        self.cb = nn.Sequential(conv, nn.BatchNorm2d(out_channels))
    def forward(self,x):
        return self.cb(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class attention_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.cbr = CBR(in_channels, out_channels, 3, stride, 1)
        self.cb = CB(out_channels, out_channels, 3, 1, 1)
        #在CB R之间引入残差单元
        self.relu = nn.ReLU(inplace=inplace)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.cbr(x)
        residual = out
        out = self.cb(out)

        #注意力机制
        out = self.ca(out) * out
        out = self.sa(out) * out

        #print(out.shape,residual.shape)
        out += residual
        out = self.relu(out)
        return out

class attention_net(nn.Module):
    def __init__(self, block, num_classes=10):
        super(attention_net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer1 = self._make_layer(block, 128, 128, 1, stride=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer2 = self._make_layer(block, 256, 256, 1, stride=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, block, in_planes, planes, num_blocks, stride):
        layers = []
        layers.append(block(in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.layer1(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = nn.MaxPool2d(2)(out)
        out = self.layer2(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = nn.AdaptiveMaxPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Attention_Resnet():
    return attention_net(attention_block)