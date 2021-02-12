from collections import OrderedDict
from logging import getLogger as get_logger
from typing import List, Mapping, Text, Any

import torch
from tasker.utils import import_reference
from torch import nn

from .classifier import SelfAdaptiveIntuitionMemoryHead
from .units import Identity
from .utils import make_divisible


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def pwconv_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Bottleneck(nn.Module):
    """
    Basic bottleneck of backbone based on MobileNet V3.

    "Andrew Howard, Ruoming Pang, Hartwig Adam, Quoc V. Le, Mark Sandler, Bo Chen, Weijun Wang, Liang-Chieh Chen, Mingxing Tan, Grace Chu, Vijay Vasudevan, Yukun Zhu: Searching for MobileNetV3. ICCV 2019: 1314-1324"
    """

    def __init__(
            self, in_channels: int, out_channels: int, kernel_size,
            stride: int, expansion: int, attention: nn.Module,
            activation: nn.Module = nn.Hardswish, conv_layer: nn.Module = nn.Conv1d,
            norm_layer: nn.Module = nn.BatchNorm1d,
            **kwargs
    ):
        """
        Initialize an instance of bottleneck.

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param kernel_size: size of convolutional kernel
        :type kernel_size: int
        :param stride: stride of convolutional operation
        :type stride: int
        :param expansion: number of expansion channels
        :type expansion: int
        :param attention: class reference of attention module
        :param activation:
        :param conv_layer: class of convolutional module
        :param norm_layer: class of normalization module
        :param anchor: whether the bottleneck produces anchor features
        :type anchor: bool
        :param stage: stage number of the bottleneck
        :type stage: int
        """
        super(Bottleneck, self).__init__()
        assert stride in (1, 2, 3, 4)
        assert kernel_size in (3, 5, 7, 9, 11, 13, 15)
        padding = (kernel_size - 1) // 2
        self.enable_residual_connection = (stride == 1 and in_channels == out_channels)
        self.first_run = True
        self._logger = get_logger('tasnem_har.models.backbone.Bottleneck')
        self.anchor = kwargs['anchor'] if 'anchor' in kwargs else False
        self.stage = kwargs['stage'] if 'stage' in kwargs else 0

        self.add_module('sequential', nn.Sequential(
            # Point-wise
            conv_layer(in_channels, expansion, 1, 1, 0, bias=False),
            norm_layer(expansion),
            activation(inplace=True),
            # Depth-wise
            conv_layer(expansion, expansion, kernel_size, stride,
                       padding, groups=expansion, bias=False),
            norm_layer(expansion),
            attention(expansion),
            activation(inplace=True),
            # Point-wise linear
            conv_layer(expansion, out_channels, 1, 1, 0, bias=False),
            norm_layer(out_channels),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        children = OrderedDict(self.named_children())

        if self.enable_residual_connection:
            result = x + children['sequential'](x)
        else:
            result = children['sequential'](x)

        if self.first_run and self.anchor:
            self._logger.info(f'Stage {self.stage} Anchor size {result.shape}')
            self.first_run = False

        return result


class TASNeModel(nn.Module):
    """
    The architecture of model produced by TASNeM-HAR. The backbone of the model is based on MobileNet V3.
    The classifier of the model is SIMH.
    """
    def __init__(
            self, in_channels: int, out_channels: int, num_classes: int,
            layers: List[Mapping[Text, Any]], conv_layer=nn.Conv1d, norm_layer=nn.BatchNorm1d,
            dropout: float = 0.0, classifier: bool = True, **kwargs
    ):
        """
        Initialize an instance of TASNeM model.

        :param in_channels: number of input raw tensor's channels
        :param out_channels: number of output features' channels
        :param num_classes: number of classes
        :param layers: profile sequences of each bottlenecks
        :param conv_layer: class of convolutional module
        :param norm_layer: class of normalization module
        :param dropout: dropout ratio
        :param classifier: whether to enable the classifier
        :param simh: parameters of the SIMH classifier
        """
        super(TASNeModel, self).__init__()
        assert conv_layer in (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        self._stage = 0

        def build_layer(layer):
            kwargs = OrderedDict(layer)
            activation = import_reference(kwargs.pop(
                'activation'
            )) if 'activation' in kwargs else nn.ReLU
            attention = import_reference(kwargs.pop(
                'attention'
            )) if 'attention' in kwargs else Identity
            kwargs['stage'] = self._stage
            if 'anchor' in kwargs and kwargs['anchor']:
                self._stage += 1

            return Bottleneck(
                in_channels=make_divisible(kwargs.pop('in_channels')),
                out_channels=make_divisible(kwargs.pop('out_channels')),
                kernel_size=kwargs.pop('kernel_size'),
                stride=kwargs.pop('stride'),
                expansion=make_divisible(kwargs.pop('expansion')),
                conv_layer=conv_layer,
                norm_layer=norm_layer,
                activation=activation,
                attention=attention,
                **kwargs
            )

        features = [
            conv_bn(
                in_channels, make_divisible(in_channels), 2,
                nlin_layer=nn.Hardswish, conv_layer=conv_layer, norm_layer=norm_layer
            )
        ]
        features.extend(map(
            build_layer,
            layers
        ))
        self.num_stages = self._stage - 1
        self.train_stage = -1

        if issubclass(conv_layer, nn.Conv1d):
            adaptive_pool_layer = nn.AdaptiveAvgPool1d
        elif issubclass(conv_layer, nn.Conv2d):
            adaptive_pool_layer = nn.AdaptiveAvgPool2d
        elif issubclass(conv_layer, nn.Conv3d):
            adaptive_pool_layer = nn.AdaptiveAvgPool3d
        else:
            raise NotImplementedError

        features.extend((
            pwconv_bn(
                layers[-1].out_channels, out_channels,
                nlin_layer=nn.Hardswish, conv_layer=conv_layer, norm_layer=norm_layer
            ),
            adaptive_pool_layer(1),
            nn.Hardswish(inplace=True)
        ))
        self.add_module('features', nn.Sequential(*features))
        if classifier:
            self.add_module('classifier', nn.Sequential(
                nn.Dropout(dropout) if abs(
                    dropout) > 1e-5 else nn.Sequential(),
                nn.Flatten(),
                SelfAdaptiveIntuitionMemoryHead(out_channels, num_classes, **
                ({} if 'simh' not in kwargs else kwargs['simh'])),
            ))
        self.has_classifier = classifier

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def stage(self, train_stage=-1):
        assert train_stage <= self.num_stages

        self.train_stage = train_stage

        return self

    def compress(self):
        if self.has_classifier:
            self.classifier = self.classifier.compress()

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.train_stage < 0 or self.train_stage == self.num_stages:
            x = self.features(x)
            if self.has_classifier:
                while len(x.shape) > 2:
                    x = x.mean(-1)
                batch_size, num_channels = x.shape
                x = self.classifier(x.view(batch_size, 1, num_channels))

        else:
            index = 0
            for layer in self.features:
                if isinstance(layer, Bottleneck):
                    if layer.stage <= self.train_stage:
                        index += 1

            x = self.features[:index + 1](x)
        return x
