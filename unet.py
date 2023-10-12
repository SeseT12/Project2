import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store the convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, config.NUM_CHANNELS)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, config.NUM_CHANNELS)

    def forward(self, x):
        # apply CONV => RELU => CONV block to the inputs and return it
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels=(config.NUM_CHANNELS, 16, 32, 64)):
        super().__init__()
        # store the encoder blocks and maxpooling layer
        self.encBlocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []
        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, encoded_features):
        # loop through the number of channels
        for i in range(len(self.channels) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            encoded_feature = self.crop(encoded_features[i], x)
            x = torch.cat([x, encoded_feature], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x

    def crop(self, encoded_features, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        (_, _, H, W) = x.shape
        encoded_features = CenterCrop([H, W])(encoded_features)
        # return the cropped features
        return encoded_features


class UNet(Module):
    def __init__(self, encoder_channels=(config.NUM_CHANNELS, 16, 32, 64, 128),
                 decoder_channels=(128, 64, 32, 16),
                 number_of_classes=3, retain_dim=True,
                 output_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encoder_channels)
        self.bridge = Block(encoder_channels[::-1][0], encoder_channels[::-1][0])
        self.decoder = Decoder(decoder_channels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decoder_channels[-1], number_of_classes, 1)
        self.retain_dim = retain_dim
        self.output_size = output_size

    def forward(self, x):
        # grab the features from the encoder
        encoder_features = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        encoder_features[::-1][0] = self.bridge(encoder_features[::-1][0])
        decoder_features = self.decoder(encoder_features[::-1][0],
                                   encoder_features[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        segmentation_map = self.head(decoder_features)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retain_dim:
            segmentation_map = F.interpolate(segmentation_map, self.output_size)
        # return the segmentation map
        return segmentation_map
