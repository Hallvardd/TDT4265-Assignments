import torch.nn as nn
import torch

# The way of building the network is inspired by the mobilnet.py from
# https://github.com/lufficc/SSD/blob/master/ssd/modeling/backbone/mobilenet.py

class FirstLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding = 1):
        super().__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=kernel_size, stride=2, padding=padding)
        )

class X2ReLUConv2d(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride_one=1, stride_two=2, padding_one=1, padding_two=1):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels= in_channels, kernel_size=kernel_size, stride=stride_one, padding=padding_one),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=kernel_size, stride=stride_two, padding=padding_two),
        )



class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        self.features = nn.ModuleList()

        # appending layers to features
        self.features.append(FirstLayer(image_channels, output_channels[0]))
        self.features.append(X2ReLUConv2d(output_channels[0], 128, output_channels[1]))
        self.features.append(X2ReLUConv2d(output_channels[1], 265, output_channels[2]))
        self.features.append(X2ReLUConv2d(output_channels[2], 128, output_channels[3]))
        self.features.append(X2ReLUConv2d(output_channels[3], 128, output_channels[4]))
        self.features.append(X2ReLUConv2d(output_channels[4], 128, output_channels[5], stride_two=1, padding_two=0))

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for layer in self.features:
            x = layer(x)
            out_features.append(x)

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

