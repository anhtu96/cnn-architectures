import torch
import torch.nn as nn

# Dictionary contains output channels for feature extractor, 'M' denotes Max-pooling layers.
CONV_CHANNELS_DICT = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
def make_layers(n_layers, has_BN=False) -> nn.Sequential:
    """ Create feature extraction layers for VGG given VGG configuration (11/13/16/19 layers and with/without BatchNorm).
    Args:
        - n_layers: number of VGG layers (int).
        - has_BN: create layers with or without BatchNorm (default: False).
    Returns:
        - a Sequential contains feature extraction layers.
    """
    layers = []
    in_channels = 3
    for out_channels in CONV_CHANNELS_DICT[n_layers]:
        if out_channels == 'M':
            layers.append(nn.MaxPool2d(2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            if has_BN:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(True))
            in_channels = out_channels
    layers = nn.Sequential(*layers)
    return layers

class VGG(nn.Module):
    def __init__(self, n_layers=16, has_BN=False, n_classes=1000):
        """ VGG class constructor.
        Args:
            - n_layers: number of layers (11/13/16/19).
            - has_BN: create Conv layers with/without BatchNorm (default: False).
            - n_classes: number of output classes.
        """
        super(VGG, self).__init__()
        self.features = make_layers(n_layers, has_BN)
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x