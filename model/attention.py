
import torch
import torch.nn as nn

class LCAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        self.conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)

    def forward(self, x):
        N, C, H, W = x.shape

        GAP = nn.AvgPool2d((H, W))
        att = GAP(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att =  att.reshape(N, C, 1, 1)
        return (x * att) + x

class LSAttention(nn.Module):
    def __init__(self, features_dim_in, reduced_channels_dim):
        super().__init__()
        self.conv1x1_1 = nn.Conv2d(features_dim_in, reduced_channels_dim, 1, 1)
        self.conv1x1_2 = nn.Conv2d(int(reduced_channels_dim*4), 1, 1, 1)
        self.dilated_conv3x3 = nn.Conv2d(reduced_channels_dim, reduced_channels_dim, 3, 1, padding=1)
        self.dilated_conv5x5 = nn.Conv2d(reduced_channels_dim, reduced_channels_dim, 3, 1, padding=2, dilation=2)
        self.dilated_conv7x7 = nn.Conv2d(reduced_channels_dim, reduced_channels_dim, 3, 1, padding=3, dilation=3)

    def forward(self, feature_maps, local_channel_output):
        att = self.conv1x1_1(feature_maps)
        d1 = self.dilated_conv3x3(att)
        d2 = self.dilated_conv5x5(att)
        d3 = self.dilated_conv7x7(att)
        att = torch.cat((att, d1, d2, d3), dim=1)
        att = self.conv1x1_2(att)
        return (local_channel_output * att) + local_channel_output

class GCAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)

    def forward(self, x):
        N, C, H, W = x.shape

        GAP = nn.AvgPool2d((H, W))
        query = key = GAP(x).reshape(N, 1, C)
        query = self.conv_q(query).sigmoid()
        key = self.conv_q(key).sigmoid().permute(0, 2, 1)
        query_key = torch.bmm(key, query).reshape(N, -1)
        query_key = query_key.softmax(-1).reshape(N, C, C)

        value = x.permute(0, 2, 3, 1).reshape(N, -1, C)
        att = torch.bmm(value, query_key).permute(0, 2, 1)
        att = att.reshape(N, C, H, W)
        return x * att


class GSAttention(nn.Module):
    def __init__(self, features_dim_in, reduced_channels_dim):
        super().__init__()
        self.conv1x1_q = nn.Conv2d(features_dim_in, reduced_channels_dim, 1, 1)
        self.conv1x1_k = nn.Conv2d(features_dim_in, reduced_channels_dim, 1, 1)
        self.conv1x1_v = nn.Conv2d(features_dim_in, reduced_channels_dim, 1, 1)
        self.conv1x1_att = nn.Conv2d(reduced_channels_dim, features_dim_in, 1, 1)

    def forward(self, feature_maps, global_channel_output):
        query = self.conv1x1_q(feature_maps)
        N, C, H, W = query.shape
        query = query.reshape(N, C, -1)
        key = self.conv1x1_k(feature_maps).reshape(N, C, -1)

        query_key = torch.bmm(key.permute(0, 2, 1), query)
        query_key = query_key.reshape(N, -1).softmax(-1)
        query_key = query_key.reshape(N, int(H*W), int(H*W))
        value = self.conv1x1_v(feature_maps).reshape(N, C, -1)
        att = torch.bmm(value, query_key).reshape(N, C, H, W)
        att = self.conv1x1_att(att)

        return (global_channel_output * att) + global_channel_output
