from torch import nn
from torch.nn import functional as F


class Identity(nn.Sequential):
    """
    A placeholder to fill in sequance module.
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()


class HardSigmoid(nn.Module):
    """
    Hard sigmoid to decrease the overhead of sigmoid in MobileNet V3.
    """
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SqueezeExcitation(nn.Module):
    """
    Squeeze-Excitation module.

    "Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu: Squeeze-and-Excitation Networks. IEEE Trans. Pattern Anal. Mach. Intell. 42(8): 2011-2023 (2020)"
    """
    def __init__(self, channel, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HardSigmoid(inplace=True)
        )

    def forward(self, x):
        b, c = x.shape[:2]
        f_s = x.shape[2:]
        if not isinstance(f_s, tuple):
            f_s = f_s,
        y = self.avg_pool(x.flatten(2, -1)).view(b, c)
        y = self.fc(y).view(b, c, *([1] * len(f_s)))
        return x * y.expand_as(x)
