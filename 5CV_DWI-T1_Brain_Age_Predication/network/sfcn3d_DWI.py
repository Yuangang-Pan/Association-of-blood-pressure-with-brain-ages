import torch
import torch.nn as nn
import torch.nn.functional as F

class SFCN(nn.Module):
    def __init__(self, in_dim=1, channel_number=[32, 64, 128, 256, 256, 64], dropout=True):
        ## DWI 91 * 109 * 91, T1 182
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = in_dim
            else:
                in_channel = channel_number[i-1]
            out_channel = channel_number[i]
            if i < n_layer-1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.regression = nn.Sequential()
        ## DWI 384, T1 1920
        dim1 = 768
        dim2 = 100
        if dropout is True:
            self.regression.add_module('dropout', nn.Dropout(0.5))
        self.regression.add_module('fl1 %d' % 1, nn.Linear(dim1, dim2))
        self.Output = nn.Linear(dim2, 1)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def feat_extract(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.regression(x)
        return x

    def forward(self, x, extra):
        feat = self.feat_extract(x)
        x = self.Output(x)
        return x, feat
