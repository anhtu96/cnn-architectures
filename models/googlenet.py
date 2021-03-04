class InceptionBlock(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        """ Inception blocks
        Args:
            - in_channels: input channels (int).
            - c1: first 1x1 conv's output channels (int).
            - c2: 3x3 reduce block's output channels (list - c2[0]: 1x1 bottleneck, c2[1]: 3x3 conv).
            - c3: 5x5 reduce block's output channels (list - c3[0]: 1x1 bottleneck, c3[1]: 5x5 conv).
            - c4: last 1x1 conv's output channels (int).
        """
        super(InceptionBlock, self).__init__()
        self.p1_1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1, stride=1),
            nn.BatchNorm2d(c1, eps=0.001))
        
        self.p2_1 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(c2[0], eps=0.001))
        self.p2_2 = nn.Sequential(
            nn.Conv2d(c2[0], c2[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c2[1], eps=0.001))

        self.p3_1 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1, stride=1),
            nn.BatchNorm2d(c3[0], eps=0.001))
        self.p3_2 = nn.Sequential(
            nn.Conv2d(c3[0], c3[1], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(c3[1], eps=0.001))

        self.p4_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.p4_2 = nn.Sequential(
            nn.Conv2d(in_channels, c4, kernel_size=1, stride=1),
            nn.BatchNorm2d(c4, eps=0.001))

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)

class GoogLeNet(nn.Module):
    """ GoogLeNet implementation.
    Check out this paper https://arxiv.org/abs/1409.4842.
    """
    def __init__(self, n_classes=1000):
        super(GoogLeNet, self).__init__()
        # for same padding, P = (F-1)/2. Source: https://cs231n.github.io/convolutional-networks/#conv
        # for valid padding, P = 0.

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2_reduce = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, (96, 128), (16, 32), 32)
        self.inception3b = InceptionBlock(256, 128, (128, 192), (32, 96), 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionBlock(480, 192, (96, 208), (16, 48), 64)
        self.inception4b = InceptionBlock(512, 160, (112, 224), (24, 64), 64)
        self.inception4c = InceptionBlock(512, 128, (128, 256), (24, 64), 64)
        self.inception4d = InceptionBlock(512, 112, (144, 288), (32, 64), 64)
        self.inception4e = InceptionBlock(528, 256, (160, 320), (32, 128), 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionBlock(832, 256, (160, 320), (32, 128), 128)
        self.inception5b = InceptionBlock(832, 384, (192, 384), (48, 128), 128)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout2d(0.2)
        self.linear = nn.Linear(1024, n_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = self.maxpool1(x)
        x = F.relu(self.conv2_reduce(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x