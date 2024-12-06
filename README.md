# Label Smoothing++ or LS++
Official Implementation of paper "Label Smoothing++: Enhanced Label Regularization for Training Neural Networks". <br>

## Introduction
### Overview
- Label Smoothing++ allows the network to learn optimal probability assignment.
- Each class learns a different probability assignment for all the non-target classes.
- The target class probability is fixed but the non-target class probabilities are flexible.
- Combined probability vectors form a C-Matrix by setting the diagonal (itself) to 0.
- One-hot vector and associated probability vectors from the C-Matrix are combined using alpha.
- Network is trained using cross-entropy loss only (using stop gradients).
- C-Matrix are trained using the reverse cross-entropy loss only (using stop gradients).

## Run command:
Run <strong>main.py</strong> to train the network with <strong>LS++</strong> with the method argument set to '<strong>lspp</strong>'. The dataset and model can be changed using the dataset and model arguments. Below is an example of training an Alexnet on CIFAR10 with LS++:<br>
```
python main.py --dataset cifar10 --model alexnet --method lspp
```

Replace cifar10 with the appropriate dataset and alexnet with the appropriate model. <br>
Supported datasets: CIFAR10, CIFAR100, FashionMNIST, SVHN, TinyImageNet, Animals10N, and ImageNet100. <br><br>
CIFAR10, CIFAR100, FashionMNIST, and SVHN datasets will be downloaded to the path specified in the "data_path" argument (default: "./data").<br>
TinyImageNet, Animals10N, and ImageNet100 need to be downloaded, and the path needs to be provided using the "data_path" argument. 

<br>
apply_wd argument controls whether weight decay should be applied to the C-Matrix. Not applying provides a sharper C-Matrix.

If you find this paper/code helpful, please cite our upcoming paper:
```
```
