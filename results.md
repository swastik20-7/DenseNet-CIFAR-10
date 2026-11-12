| Run | Model | Depth | k | theta | Params (M) | Epochs | Aug | Seeds | Final test err (%) | Best-epoch err (%) | Train min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abl_k6 | DenseNet-BC | 40 | 6 | 0.5 | 0.046 | 15 | yes | 1 | 18.04 | 18.00 | 6.2 |
| abl_k12 | DenseNet-BC | 40 | 12 | 0.5 | 0.176 | 15 | yes | 1 | 14.54 | 14.41 | 6.6 |
| abl_k24 | DenseNet-BC | 40 | 24 | 0.5 | 0.690 | 15 | yes | 1 | 11.53 | 11.53 | 10.9 |
| abl_t0.25 | DenseNet-BC | 40 | 12 | 0.25 | 0.150 | 15 | yes | 1 | 15.54 | 15.42 | 6.6 |
| abl_t0.5 | DenseNet-BC | 40 | 12 | 0.5 | 0.176 | 15 | yes | 1 | 14.93 | 14.72 | 6.7 |
| abl_t1.0 | DenseNet-B | 40 | 12 | 1.0 | 0.238 | 15 | yes | 1 | 14.04 | 13.79 | 6.7 |
| densenet-BC_L40_k12_t0.5 | DenseNet-BC | 40 | 12 | 0.5 | 0.176 | 20 | yes | 1 | 12.90 | 12.82 | 8.7 |
| densenet-BC_L58_k12_t0.5 | DenseNet-BC | 58 | 12 | 0.5 | 0.314 | 20 | yes | 1 | 12.32 | 12.25 | 11.5 |
| densenet-BC_L100_k12_t0.5 | DenseNet-BC | 100 | 12 | 0.5 | 0.769 | 20 | yes | 1 | 10.31 | 10.25 | 22.9 |
| preact-resnet_L20 | ResNet (pre-act) | 20 | - | - | 0.270 | 20 | yes | 1 | 12.66 | 12.66 | 6.9 |
| preact-resnet_L32 | ResNet (pre-act) | 32 | - | - | 0.464 | 20 | yes | 1 | 11.85 | 11.71 | 7.6 |
| preact-resnet_L56 | ResNet (pre-act) | 56 | - | - | 0.853 | 20 | yes | 1 | 11.00 | 11.00 | 9.4 |

*Final test err = error after the last epoch (mean ± std over seeds if >1). Best-epoch err is picked on the test set and is only informational.*
