import torch.nn as nn


def conv_layer(in_channels, out_channels,
               kernel: int = 3,
               stride: int = 1,
               padding: int = 1,
               dropout_rate: float = 0.0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=(kernel, kernel),
                  stride=(stride, stride),
                  padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, stride=(2, 2)),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=dropout_rate)
    )


def fully_conn(in_features, out_features, dropout_rate: float = 0.0):
    return nn.Sequential(
        nn.Linear(in_features=in_features,
                  out_features=out_features),
        nn.ReLU(inplace=False),
        nn.Dropout(p=dropout_rate)
    )


def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight)  # .xavier_normal_(layer.weight)
    elif isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)


class ImageClassifier(nn.Module):

    def __init__(self, num_classes=3):
        super(ImageClassifier, self).__init__()
        dropout = 0.2
        in_ch = 16
        self.conv_down1 = conv_layer(3, in_ch,
                                     kernel=5,
                                     stride=1,
                                     padding=2,
                                     dropout_rate=dropout)
        self.conv_down1.apply(init_weights)
        dropout *= 1.2
        self.conv_down2 = conv_layer(in_ch, 2 * in_ch,
                                     kernel=5,
                                     stride=1,
                                     padding=2,
                                     dropout_rate=dropout)
        self.conv_down2.apply(init_weights)
        dropout *= 1.2
        self.conv_down3 = conv_layer(2 * in_ch, 4 * in_ch,
                                     kernel=3,
                                     stride=2,
                                     padding=1,
                                     dropout_rate=dropout)
        self.conv_down3.apply(init_weights)

        num_features = 32
        self.fully_conn1 = fully_conn(in_features=14 * 14 * 4 * in_ch, out_features=num_features, dropout_rate=0.2)
        self.fully_conn2 = fully_conn(in_features=num_features, out_features=num_features, dropout_rate=0.2)

        self.fully_conn1.apply(init_weights)
        self.fully_conn2.apply(init_weights)

        self.num_classes = num_classes
        self.fully_conn3 = nn.Linear(in_features=num_features, out_features=num_classes)
        self.fully_conn3.apply(init_weights)

    def forward(self, x):
        x = self.conv_down1(x)
        x = self.conv_down2(x)
        x = self.conv_down3(x)
        x = nn.Flatten()(x)
        x = self.fully_conn1(x)
        x = self.fully_conn2(x)
        out = self.fully_conn3(x)
        return out
