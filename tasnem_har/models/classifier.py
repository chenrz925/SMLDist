from collections import OrderedDict
from copy import deepcopy
from logging import getLogger as get_logger

import torch
from torch import nn

from .hopfield import Hopfield


class SelfAdaptiveIntuitionMemoryHead(nn.Module):
    """
    Self-adaptive Intuition-Memory Head (SIMH)
    """

    def __init__(self, in_features, out_features, **kwargs):
        """
        :param in_features:
        :type in_features: int
        :param out_features:
        :type out_features: int
        :param update_steps_max:
        :type update_steps_max: int
        :param update_steps_eps:
        :type update_steps_eps: float
        :param eval_mode:
        :type eval_mode: str
        """
        self._logger = get_logger('tasnem_har.models.classifier.SelfAdaptiveIntuitionMemoryHead')
        self.in_features = in_features
        self.out_features = out_features
        self.update_steps_max = kwargs['update_steps_max'] if 'update_steps_max' in kwargs else 2
        self.update_steps_eps = kwargs['update_steps_eps'] if 'update_steps_eps' in kwargs else 1e-2
        self.eval_mode = kwargs['eval_mode'] if 'eval_mode' in kwargs else 'M'

        super(SelfAdaptiveIntuitionMemoryHead, self).__init__()
        self.add_module('memory', Hopfield(
            in_features, in_features, out_features,
            update_steps_max=self.update_steps_max, update_steps_eps=self.update_steps_eps,
        ))
        self.add_module('intuition', nn.Linear(in_features, out_features))
        self.register_parameter('weight', nn.Parameter(
            torch.tensor([0.0, 0.0]), requires_grad=True))
        self._logger.info(f'Update steps max {self.update_steps_max}')
        self._logger.info(f'Update steps eps {self.update_steps_eps}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        children = OrderedDict(self.named_children())
        parameters = OrderedDict(self.named_parameters())

        batch_size, num_channels = x.shape
        y_h = children['memory'](x.view(batch_size, 1, num_channels))
        y_l = children['intuition'](x).view(batch_size, 1, self.out_features)
        y_c = torch.cat((y_h, y_l), dim=1)

        y_w = parameters['weight'].softmax(0).view(1, 2, 1).expand(
            batch_size, 2, self.out_features) * y_c.softmax(-1)
        y_f = y_w[:, 0, :] + y_w[:, 1, :]
        if self.training:
            return y_f
        elif hasattr(self, 'eval_mode'):
            if self.eval_mode == 'I':
                return y_l[:, 0, :]
            elif self.eval_mode == 'M':
                return y_h[:, 0, :]
            else:
                return y_f
        else:
            return y_f

    def compress(self):
        """
        Compress the model by removing the redundant intuition unit.
        :return: compressed SIMH.
        """
        return _SelfAdaptiveIntuitionMemoryHeadPruned(self)


class _SelfAdaptiveIntuitionMemoryHeadPruned(nn.Module):
    def __init__(self, model: SelfAdaptiveIntuitionMemoryHead):
        super(_SelfAdaptiveIntuitionMemoryHeadPruned, self).__init__()

        self.add_module('memory', deepcopy(model.memory))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels = x.shape

        return self.memory(x.view(batch_size, 1, num_channels)).softmax(-1).flatten(1)
