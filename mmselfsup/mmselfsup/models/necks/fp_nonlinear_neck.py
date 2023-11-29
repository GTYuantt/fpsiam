# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_norm_layer, ConvModule
from mmcv.runner import BaseModule
from .fpn import FPN

from ..builder import NECKS


@NECKS.register_module()
class FPNonLinearNeck(BaseModule):
    """The feature pyramid non-linear neck.

    Structure: conv-fc-bn-[relu-fc-bn] where the substructure in [] can be repeated.
    For the default setting, the repeated time is 1.
    The neck can be used in many algorithms, e.g., FPBYOL, FPSiam.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 fpn_out_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_bias=False,
                 with_last_bn=True,
                 with_last_bn_affine=True,
                 with_last_bias=False,
                 with_avg_pool=True,
                 vit_backbone=False,
                 norm_cfg=dict(type='SyncBN'),
                 init_cfg=[
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(FPNonLinearNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.fpn_out_channels = fpn_out_channels
        self.with_avg_pool = with_avg_pool
        self.vit_backbone = vit_backbone
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        # self.conv0 = nn.ModuleList()
        # for i in range(0, len(in_channels)):
        #     conv0_ = ConvModule(
        #             in_channels[i],
        #             hid_channels,
        #             1,
        #             inplace=False)
        #     self.conv0.append(conv0_)

        self.FPN = FPN(in_channels=self.in_channels,
                       out_channels=self.fpn_out_channels)

        self.fc0 = nn.Linear(fpn_out_channels, hid_channels, bias=with_bias)
        self.bn0 = build_norm_layer(norm_cfg, hid_channels)[1]

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(hid_channels, this_channels, bias=with_bias))
                self.add_module(f'bn{i}',
                                build_norm_layer(norm_cfg, this_channels)[1])
                self.bn_names.append(f'bn{i}')
            else:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(
                        hid_channels, this_channels, bias=with_last_bias))
                if with_last_bn:
                    self.add_module(
                        f'bn{i}',
                        build_norm_layer(
                            dict(**norm_cfg, affine=with_last_bn_affine),
                            this_channels)[1])
                    self.bn_names.append(f'bn{i}')
                else:
                    self.bn_names.append(None)
            self.fc_names.append(f'fc{i}')

    def forward(self, x):
        assert len(x) == len(self.in_channels)
        out = []
        x = self.FPN(x)

        # global feature map
        x_g = x[1]
        if self.vit_backbone:
            x_g = x_g[-1]
        if self.with_avg_pool:
            x_g = self.avgpool(x_g)
        x_g = x_g.view(x_g.size(0), -1)
        x_g = self.fc0(x_g)
        x_g = self.bn0(x_g)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x_g = self.relu(x_g)
            x_g = fc(x_g)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x_g = bn(x_g)
        out.append(x_g)

        # local feature map
        x_l = x[0]
        if self.vit_backbone:
            x_l = x_l[-1]
        if self.with_avg_pool:
            x_l = self.avgpool(x_l)
        x_l = x_l.view(x_l.size(0), -1)
        x_l = self.fc0(x_l)
        x_l = self.bn0(x_l)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x_l = self.relu(x_l)
            x_l = fc(x_l)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x_l = bn(x_l)
        out.append(x_l)

        return tuple(out)
