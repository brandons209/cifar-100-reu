'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
      }


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):

        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        print('%s Model Successfully Built \n' % self.vgg_name)

        return nn.Sequential(*layers)


class Custom_Model(nn.Module):
    def __init__(self, num_classes, num_blocks):
        super(Custom_Model, self).__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.features, out_channels = self.build_blocks()
        self.classifier = self.build_classifier(out_channels)

    def forward(self, x):
        out = self.features(x)
        # 1d vector in channel dimensions, need to move that to the width dimension
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def build_classifier(self, out_channels):
        layers = [nn.Linear(out_channels, 2048), nn.ReLU(inplace=True), nn.Dropout(p=0.3, inplace=True)]
        layers.append(nn.Linear(2048, self.num_classes))

        return nn.Sequential(*layers)

    def build_blocks(self):
        layers = []
        input_channels = 3
        out_channels = 32
        for i in range(self.num_blocks):
            padding = (((out_channels - 1) * 2) - input_channels + 2) // 2
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=2, stride=2, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            input_channels = out_channels
            padding = (((out_channels - 1) * 2) - input_channels + 2) // 2
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=2, stride=2, padding=padding))
            nn.ReLU(inplace=True)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            out_channels *= 2

        layers.append(nn.AvgPool2d(kernel_size=out_channels))

        return nn.Sequential(*layers), out_channels

from torchsummary import summary
summary(Custom_Model(100, 6), (3, 32, 32))
