"""
train.py -- train DenseNet(-B/-C/-BC) on CIFAR-10 (paper's setup, reduced epochs).

Paper setup: SGD, Nesterov momentum 0.9, weight decay 1e-4, batch 64, 300 epochs,
lr 0.1 divided by 10 at 50% and 75% of training, dropout 0.2 only without augmentation.

Examples:
    # main model, reduced budget (Colab T4)
    python train.py --depth 100 --growth 12 --theta 0.5 --bottleneck --epochs 40

    # ablation run (small, fast)
    python train.py --depth 40 --growth 12 --theta 0.5 --bottleneck --epochs 30 --name abl_k12_t05

    # plain DenseNet (no B, no C)
    python train.py --depth 40 --growth 12 --theta 1.0 --epochs 30

    # ResNet baseline (same pipeline, same hyper-parameters)
    python train.py --arch resnet --depth 56 --epochs 40 --amp

    # smoke test without downloading data (random images)
    python train.py --depth 22 --growth 12 --fake --epochs 2
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from densenet import DenseNet, count_params
from resnet import CifarResNet

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_args():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--arch", choices=["densenet", "resnet", "plain"], default="densenet",
                   help="densenet | resnet (CIFAR ResNet, depth=6n+2) | plain (ResNet without shortcuts)")
    p.add_argument("--preact", action="store_true", help="pre-activation ResNet (arch resnet/plain)")
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--growth", type=int, default=12, help="growth rate k")
    p.add_argument("--theta", type=float, default=0.5, help="compression factor (1.0 = none)")
    p.add_argument("--bottleneck", action="store_true", help="use 1x1 bottleneck (B)")
    p.add_argument("--drop-rate", type=float, default=None,
                   help="dropout; default 0.0 with augmentation, 0.2 with --no-aug (as in paper)")
    # training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--no-aug", action="store_true", help="disable crop+flip (C10 instead of C10+)")
    p.add_argument("--amp", action="store_true", help="mixed precision (faster on T4)")
    p.add_argument("--seed", type=int, default=0)
    # io
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out", type=str, default="./runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--fake", action="store_true", help="random data, for smoke tests only")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint in run dir")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(args):
    norm = T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if args.no_aug:
        train_tf = T.Compose([T.ToTensor(), norm])
    else:
        train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), norm])
    test_tf = T.Compose([T.ToTensor(), norm])

    if args.fake:
        train_set = torchvision.datasets.FakeData(512, (3, 32, 32), 10, train_tf)
        test_set = torchvision.datasets.FakeData(256, (3, 32, 32), 10, test_tf)
    else:
        train_set = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=test_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_set, 256, shuffle=False, num_workers=args.workers,
                             pin_memory=pin, persistent_workers=args.workers > 0)
    return train_loader, test_loader


def run_epoch(model, loader, device, optimizer=None, scaler=None, amp=False):
    """One pass over `loader`. Trains if optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, 100.0 * correct / n


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"

    drop = args.drop_rate if args.drop_rate is not None else (0.2 if args.no_aug else 0.0)
    if args.arch == "densenet":
        model = DenseNet(args.depth, args.growth, args.bottleneck, args.theta, drop, 10)
        variant = ("BC" if args.bottleneck and args.theta < 1 else
                   "B" if args.bottleneck else "C" if args.theta < 1 else "plain")
        default_name = f"densenet-{variant}_L{args.depth}_k{args.growth}_t{args.theta}_s{args.seed}"
    else:
        model = CifarResNet(args.depth, 10, residual=(args.arch == "resnet"), preact=args.preact)
        variant = ("preact-" if args.preact else "") + args.arch
        default_name = f"{variant}_L{args.depth}_s{args.seed}"
    model = model.to(device)
    n_params = count_params(model)
    name = args.name or default_name
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run: {name} | device: {device} | params: {n_params/1e6:.3f}M | "
          f"aug: {not args.no_aug} | dropout: {drop} | amp: {amp}")

    train_loader, test_loader = get_loaders(args)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.wd, nesterov=True)
    # lr / 10 at 50% and 75% of the total epochs (paper)
    milestones = [int(0.5 * args.epochs), int(0.75 * args.epochs)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    start_epoch, best_acc, history = 0, 0.0, []
    ckpt_path = os.path.join(run_dir, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc, history = ck["epoch"] + 1, ck["best_acc"], ck["history"]
        print(f"Resumed from epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        te = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer, scaler, amp)
        te_loss, te_acc = run_epoch(model, test_loader, device, amp=amp)
        scheduler.step()
        best_acc = max(best_acc, te_acc)
        history.append(dict(epoch=epoch + 1, lr=optimizer.param_groups[0]["lr"],
                            train_loss=tr_loss, train_acc=tr_acc,
                            test_loss=te_loss, test_acc=te_acc))
        print(f"[{epoch+1:3d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.2f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.2f} (err {100-te_acc:.2f}) | "
              f"{time.time()-te:.0f}s")
        torch.save(dict(model=model.state_dict(), opt=optimizer.state_dict(),
                        sched=scheduler.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch, best_acc=best_acc, history=history), ckpt_path)

    # ---- save logs + summary ----
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    final = history[-1]
    summary = dict(name=name, arch=args.arch, variant=variant, depth=args.depth, growth_rate=args.growth,
                   theta=args.theta, bottleneck=args.bottleneck, params=n_params,
                   epochs=args.epochs, batch_size=args.batch_size, augmentation=not args.no_aug,
                   dropout=drop, seed=args.seed,
                   final_test_acc=final["test_acc"], final_test_err=100 - final["test_acc"],
                   best_test_acc=best_acc, best_test_err=100 - best_acc,
                   train_minutes=(time.time() - t0) / 60)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    print(f"\nReport the FINAL-epoch error ({summary['final_test_err']:.2f}%), "
          f"not the best-epoch one, to match the paper's protocol.")


if __name__ == "__main__":
    main()
