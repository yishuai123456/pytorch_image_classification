import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..initializer import create_initializer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 remove_first_relu,
                 add_last_bn,
                 preact=False):
        super().__init__()

        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        if add_last_bn:
            self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x),
                       inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            # preactivation only for residual path
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)

        if self._add_last_bn:
            y = self.bn3(y)

        y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 remove_first_relu,
                 add_last_bn,
                 preact=False):
        super().__init__()

        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact

        bottleneck_channels = out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               bottleneck_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        if add_last_bn:
            self.bn4 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x),
                       inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            # preactivation only for residual path
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)

        if self._add_last_bn:
            y = self.bn4(y)

        y += self.shortcut(x)
        return y


class Resnet(nn.Module):
    def __init__(self, config, input_channels):
        super().__init__()

        model_config = config.model.resnet_preact
        initial_channels = model_config.initial_channels
        self._remove_first_relu = model_config.remove_first_relu
        self._add_last_bn = model_config.add_last_bn
        block_type = model_config.block_type
        depth = model_config.depth
        preact_stage = model_config.preact_stage

        assert block_type in ['basic', 'bottleneck']
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            assert n_blocks_per_stage * 6 + 2 == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert n_blocks_per_stage * 9 + 2 == depth

        n_channels = [
            initial_channels,
            initial_channels * 2 * block.expansion,
            initial_channels * 4 * block.expansion,
        ]

        self.conv = nn.Conv2d(input_channels,
                              n_channels[0],
                              kernel_size=(3, 3),
                              stride=1,
                              padding=1,
                              bias=False)

        self.stage1 = self._make_stage(n_channels[0],
                                       n_channels[0],
                                       n_blocks_per_stage,
                                       block,
                                       stride=1,
                                       preact=preact_stage[0])
        self.stage2 = self._make_stage(n_channels[0],
                                       n_channels[1],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2,
                                       preact=preact_stage[1])
        self.stage3 = self._make_stage(n_channels[1],
                                       n_channels[2],
                                       n_blocks_per_stage,
                                       block,
                                       stride=2,
                                       preact=preact_stage[2])
        self.bn = nn.BatchNorm2d(n_channels[2])

        # compute conv feature size
        with torch.no_grad():
            dummy_data = torch.zeros(
                (1, input_channels, config.dataset.image_size,
                 config.dataset.image_size),
                dtype=torch.float32)
            self.feature_size = self._forward_conv(dummy_data).view(
                -1).shape[0]

        self.fc = nn.Linear(self.feature_size, config.environment.fc)



    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride,
                    preact):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = f'block{index + 1}'
            if index == 0:
                stage.add_module(
                    block_name,
                    block(in_channels,
                          out_channels,
                          stride=stride,
                          remove_first_relu=self._remove_first_relu,
                          add_last_bn=self._add_last_bn,
                          preact=preact))
            else:
                stage.add_module(
                    block_name,
                    block(out_channels,
                          out_channels,
                          stride=1,
                          remove_first_relu=self._remove_first_relu,
                          add_last_bn=self._add_last_bn,
                          preact=False))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x),
                   inplace=True)  # apply BN and ReLU before average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.factor_nums = config.environment.nums
        self.channels = config.environment.channels
        self.resnets=nn.ModuleList( [Resnet(config, num_channels) for num_channels in config.environment.channels])

        self.fc0= nn.Linear(config.environment.n_env_info, config.environment.n_env_info)
        self.fc1 = nn.Linear(self.factor_nums * config.environment.fc, config.environment.fc)
        self.fc2 = nn.Linear( config.environment.fc+config.environment.n_env_info, int(config.environment.fc/2))
        self.fc3=nn.Linear(int(config.environment.fc/2),config.dataset.n_classes)

        # initialize weights
        initializer = create_initializer(config.model.init_mode)
        self.apply(initializer)
    def forward(self, x,env_info):

        channel_start = 0
        channel_end = 0
        y = []
        for i,resnet in enumerate(self.resnets):
            channel_end = self.channels[i] + channel_end
            out = resnet(x[:, channel_start:channel_end,:,:])
            out = F.relu(out)
            channel_start = channel_end
            y.append(out)
        y = torch.cat(y, dim=1)
        x = F.relu(self.fc1(y))
        #print(x)
        #env_info = F.relu(self.fc0(env_info))
        #print(env_info)
        x= torch.cat((x,env_info),dim=1)

        x = self.fc2(x)
        x = self.fc3(F.relu(x))
        #print(x)
        return x
