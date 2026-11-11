import torch
import torch.nn as nn
from torchvision import models


class SkySegNet(torch.nn.Module):
    '''
    This network predicts foreground/background eegion in a given picture. This network follows
    Unet like architecture where downsampling part is pre-trained ResNet layers and Upsampling part
    is based on FPN like decoder.
    '''
    def __init__(self, in_channels=3, out_channels=1):
        super(SkySegNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)
        self.ups = nn.ModuleList() #List of upsampling layers
        self.features = [64, 128, 256, 512]
        for feature in reversed(self.features[1:]):
            self.ups.append(
                nn.Conv2d(feature*4, feature*2, kernel_size=3, padding=1)
            )
        self.ups.append(
            nn.Conv2d(self.features[0]*4, self.features[0], kernel_size=3, padding=1)
        )

        self.pred1 = nn.Conv2d(self.features[0], self.features[0], kernel_size=3, padding=1)
        self.pred2 = nn.Conv2d(self.features[0], out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        skip_connections = []
        x = self.resnet.layer1(x) #in=64, out=64
        skip_connections.append(x)
        x = self.resnet.layer2(x) #in=64, out=128
        skip_connections.append(x)
        x = self.resnet.layer3(x) #in=128, out=256
        skip_connections.append(x)
        x = self.resnet.layer4(x) #in=256, out=512
        skip_connections.append(x)

        up_layers = len(self.ups)
        for i in range(up_layers):
            if i != 0:
                x = x + skip_connections[up_layers-i-1]
            x = self.upsample(self.relu(self.ups[i](x)))

        # output layers
        x = self.upsample(self.relu(self.pred1(x)))
        x = self.sigmoid(self.pred2(x))
        return x