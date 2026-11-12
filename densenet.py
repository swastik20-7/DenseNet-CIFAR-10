"""
densenet.py -- DenseNet / DenseNet-B / DenseNet-C / DenseNet-BC for CIFAR.

Written from scratch following:
    Huang, Liu, van der Maaten, Weinberger.
    "Densely Connected Convolutional Networks", CVPR 2017.

Key equation (paper Eq. 2):   x_l = H_l([x_0, x_1, ..., x_{l-1}])

Variants (controlled by two flags):
    bottleneck=False, theta=1.0  -> DenseNet     (plain)
    bottleneck=True,  theta=1.0  -> DenseNet-B   (1x1 bottleneck conv, 4k channels)
    bottleneck=False, theta<1.0  -> DenseNet-C   (compression at transitions)
    bottleneck=True,  theta<1.0  -> DenseNet-BC  (both)  <- main model of the project
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """H_l(.) : BN-ReLU-Conv3x3  (or BN-ReLU-Conv1x1-BN-ReLU-Conv3x3 if bottleneck).

    Takes the concatenation of ALL previous feature-maps, produces only k new
    feature-maps, and returns [input, new] (this concatenation is the dense connection).
    """

    def __init__(self, in_ch, growth_rate, bottleneck=True, drop_rate=0.0):
        super().__init__()
        self.bottleneck = bottleneck
        self.drop_rate = drop_rate
        if bottleneck:
            inter = 4 * growth_rate                      # paper: 1x1 conv outputs 4k maps
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.conv1 = nn.Conv2d(in_ch, inter, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(inter)
            self.conv2 = nn.Conv2d(inter, growth_rate, kernel_size=3, padding=1, bias=False)
        else:
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.conv2 = nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        if self.bottleneck:
            out = self.conv1(F.relu(self.bn1(x)))
            if self.drop_rate > 0:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
            out = self.conv2(F.relu(self.bn2(out)))
        else:
            out = self.conv2(F.relu(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)                # dense connection: CONCATENATE


class DenseBlock(nn.Module):
    """Stack of `num_layers` DenseLayers. Layer i sees k0 + k*i input channels."""

    def __init__(self, num_layers, in_ch, growth_rate, bottleneck, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_ch + i * growth_rate, growth_rate, bottleneck, drop_rate))
        self.layers = nn.Sequential(*layers)
        self.out_channels = in_ch + num_layers * growth_rate

    def forward(self, x):
        return self.layers(x)


class Transition(nn.Module):
    """BN - ReLU - Conv1x1 (compress to floor(theta*m) maps) - AvgPool2x2."""

    def __init__(self, in_ch, theta=0.5, drop_rate=0.0):
        super().__init__()
        self.out_channels = int(math.floor(theta * in_ch))
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, self.out_channels, kernel_size=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseNet(nn.Module):
    """DenseNet for 32x32 inputs (CIFAR / SVHN): 3 dense blocks, equal number of layers.

    Args:
        depth:       total depth L. Layers per block = (L-4)/6 if bottleneck else (L-4)/3.
        growth_rate: k.
        bottleneck:  use 1x1 bottleneck (the "B" in DenseNet-B/-BC).
        theta:       compression factor in (0, 1]; <1 gives the "C".
        drop_rate:   dropout after each conv (paper uses 0.2 only WITHOUT data augmentation).
        num_classes: 10 for CIFAR-10.
    """

    def __init__(self, depth=100, growth_rate=12, bottleneck=True, theta=0.5,
                 drop_rate=0.0, num_classes=10):
        super().__init__()
        assert 0 < theta <= 1.0, "theta must be in (0, 1]"
        divisor = 6 if bottleneck else 3
        assert (depth - 4) % divisor == 0, \
            f"(depth-4) must be divisible by {divisor} (got depth={depth})"
        n = (depth - 4) // divisor                      # layers per dense block

        # Initial conv: 2k channels for DenseNet-BC, otherwise 16 (paper, Sec. 3)
        ch = 2 * growth_rate if (bottleneck and theta < 1.0) else 16
        self.conv0 = nn.Conv2d(3, ch, kernel_size=3, padding=1, bias=False)

        blocks = []
        for b in range(3):
            block = DenseBlock(n, ch, growth_rate, bottleneck, drop_rate)
            blocks.append(block)
            ch = block.out_channels
            if b != 2:                                   # transition between blocks only
                trans = Transition(ch, theta, drop_rate)
                blocks.append(trans)
                ch = trans.out_channels
        self.features = nn.Sequential(*blocks)

        self.bn_final = nn.BatchNorm2d(ch)
        self.fc = nn.Linear(ch, num_classes)
        self.final_channels = ch
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv0(x)
        x = self.features(x)
        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)      # global average pooling
        return self.fc(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def densenet_bc(depth=100, growth_rate=12, theta=0.5, **kw):
    return DenseNet(depth, growth_rate, bottleneck=True, theta=theta, **kw)


if __name__ == "__main__":
    # Quick sanity check: shapes + parameter counts (compare with paper's Table 2).
    #   DenseNet-BC L=100 k=12 -> ~0.8M     DenseNet L=40 k=12 -> ~1.0M
    #   DenseNet L=100 k=12    -> ~7.0M     DenseNet-BC L=250 k=24 -> ~15.3M
    cfgs = [("DenseNet     L=40  k=12", dict(depth=40, growth_rate=12, bottleneck=False, theta=1.0)),
            ("DenseNet     L=100 k=12", dict(depth=100, growth_rate=12, bottleneck=False, theta=1.0)),
            ("DenseNet-BC  L=100 k=12", dict(depth=100, growth_rate=12, bottleneck=True, theta=0.5)),
            ("DenseNet-BC  L=250 k=24", dict(depth=250, growth_rate=24, bottleneck=True, theta=0.5))]
    for name, cfg in cfgs:
        m = DenseNet(**cfg)
        y = m(torch.randn(2, 3, 32, 32))
        print(f"{name}: params = {count_params(m)/1e6:.2f}M, output = {tuple(y.shape)}")
