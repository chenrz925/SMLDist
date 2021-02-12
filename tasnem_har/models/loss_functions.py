import torch
from torch.nn import functional as F
from torch import nn


class HintonCrossEntropyLoss(nn.Module):
    def __init__(self, teacher: nn.Module, ratio: float = 1.0, temperature: float = 1.0, reduction: str = 'mean'):
        super(HintonCrossEntropyLoss, self).__init__()

        self.ratio = ratio
        self.temperature = temperature
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

        self.add_module('teacher', teacher)

        teacher.eval()

    def forward(self, x: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        children = dict(self.named_children())
        num_classes = y_pred.shape[-1]

        with torch.no_grad():
            y_soft: torch.Tensor = children['teacher'](x)

        y_pred_t = (y_pred / self.temperature).softmax(-1)
        y_pred = y_pred.softmax(-1)
        y = F.one_hot(y, num_classes)
        y_soft_t = (y_soft / self.temperature).softmax(-1)

        return self._cross_entropy(y, y_pred) + self.ratio * self._cross_entropy(y_soft_t, y_pred_t)

    def _cross_entropy(self, y: torch.Tensor, y_pred: torch.Tensor):
        raw = -(y * y_pred.log()).sum(-1)
        if self.reduction == 'mean':
            return raw.mean()
        elif self.reduction == 'sum':
            return raw.sum()
        else:
            return raw


class LogitsDistillingLoss(HintonCrossEntropyLoss):
    """
    The loss function to distill logits in SKD.
    """
    def __init__(self, teacher, **kwargs):
        """
        Initialize an instance of logits distilling loss.

        :param teacher: teacher model to provide soft target
        """
        super(LogitsDistillingLoss, self).__init__(teacher)

        self.hardness_factor = 1 if 'hardness_factor' not in kwargs else kwargs[
            'hardness_factor']

    def forward(self, x: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        children = dict(self.named_children())
        num_classes = y_pred.shape[-1]

        with torch.no_grad():
            y_soft: torch.Tensor = children['teacher'](x)

        y_pred = (y_pred / y_pred.norm(dim=-1, keepdim=True)).softmax(-1)
        y_hard = F.one_hot(y, num_classes).type(torch.float32)
        y_soft = (y_soft / y_soft.norm(dim=-1, keepdim=True)).softmax(-1)

        y_soft[y_soft.argmax(
            dim=-1) != y] = y_hard[y_soft.argmax(dim=-1) != y] * self.hardness_factor
        y_soft = y_soft.softmax(dim=-1)
        y_pred = y_pred.softmax(dim=-1)
        y_hard[y_hard > 0.5] = 1.0 / self.temperature
        y_hard[y_hard < 0.5] = 1.0 - 1.0 / self.temperature

        return self._cross_entropy(y_hard, y_pred) + self.ratio * self._cross_entropy(y_soft, y_pred)
