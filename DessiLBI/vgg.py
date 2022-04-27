import torch.nn as nn
import torch.nn.functional as F

def init_weights_(m):
    """
    Initializes weights of m according to Xavier normal method.

    :param m: module
    :return:
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class VGG_A(nn.Module):
    """VGG_A_BatchNorm model

    size of Linear layers is smaller since input assumed to be 32x32x3, instead of
    224x224x3
    """

    def __init__(self,n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.down1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        
        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.down2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128,256,3,1,1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU()
        self.down3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(4*4*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, self.n_classes),
        )

    def forward(self,x):
        n_batch = x.shape[0]
        
        #conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.down1(x)   

        #conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.down2(x)

        #conv3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)

        residual = x 
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x += residual
        x = self.relu3_2(x)
        x = self.down3(x)

        x = x.view(n_batch,-1)
        y = self.classifier(x)
        return F.log_softmax(y, dim=1)