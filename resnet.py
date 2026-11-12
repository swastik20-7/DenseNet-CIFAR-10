"""
resnet.py -- CIFAR ResNet baseline, written from scratch.

Follows the CIFAR architecture of He et al., "Deep Residual Learning for Image
Recognition" (CVPR 2016), Sec. 4.2:
    depth = 6n + 2  ->  20, 32, 44, 56, 110 ...
    3x3 conv (16) -> 3 stages of n BasicBlocks with 16/32/64 channels
    -> global average pool -> FC.
    Downsampling by stride 2 at the first block of stages 2 and 3.
    Shortcut "option A": identity with stride-2 subsampling + zero-padded channels
    (parameter-free, as in the paper, so ResNet-20 has ~0.27M and ResNet-110 ~1.7M params).

Flags:
    residual=False -> PlainNet: same layers WITHOUT the shortcut. Used to reproduce the
                      "degradation problem" (paper Fig. 6) in your own experiments.
    preact=True    -> pre-activation ResNet (He et al., ECCV 2016, ref [12] in DenseNet paper),
                      BN-ReLU-Conv ordering; this is the ResNet variant the DenseNet paper
                      compares against in Fig. 4 (middle).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionAShortcut(nn.Module):
    """Parameter-free shortcut: spatial stride-2 subsample + zero-pad extra channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pad = (out_ch - in_ch) // 2

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad))


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, residual=True, preact=False):
        super().__init__()
        self.residual, self.preact = residual, preact
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        # preact: bn1 normalises the block INPUT; post-act: bn1 normalises conv1's output
        self.bn1 = nn.BatchNorm2d(in_ch if preact else out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch if preact else out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = OptionAShortcut(in_ch, out_ch)

    def forward(self, x):
        if self.preact:
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + (x if self.shortcut is None else self.shortcut(x))   # y = F(x) + x
        if not self.preact:
            out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, depth=20, num_classes=10, residual=True, preact=False):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2 (20, 32, 44, 56, 110, ...)"
        n = (depth - 2) // 6
        self.preact = preact
        self.conv0 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(16) if not preact else None

        layers, in_ch = [], 16
        for stage, out_ch in enumerate([16, 32, 64]):
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1
                layers.append(BasicBlock(in_ch, out_ch, stride, residual, preact))
                in_ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.bn_final = nn.BatchNorm2d(64) if preact else None
        self.fc = nn.Linear(64, num_classes)
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
        if not self.preact:
            x = F.relu(self.bn0(x))
        x = self.layers(x)
        if self.preact:
            x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Expected (paper, Sec. 4.2): ResNet-20 0.27M, 32 0.46M, 44 0.66M, 56 0.85M, 110 1.7M
    for d in [20, 32, 44, 56, 110]:
        m = CifarResNet(d)
        y = m(torch.randn(2, 3, 32, 32))
        print(f"ResNet-{d:<3d} params = {count_params(m)/1e6:.2f}M, output = {tuple(y.shape)}")
    print(f"PlainNet-56 params = {count_params(CifarResNet(56, residual=False))/1e6:.2f}M")
    print(f"PreAct ResNet-56 params = {count_params(CifarResNet(56, preact=True))/1e6:.2f}M")
