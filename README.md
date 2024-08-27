# Label Smoothing++ (LS++)
This is the official PyTorch Implementation of our upcoming BMVC 2024 PatchRot paper "Label Smoothing++: Enhanced Label Regularization for Training Neural Networks". <br>

## Introduction
 <br>

## Run command:
Run <strong>main.py</strong> to train the network with LS++ with the method argument set to 'lspp'. The dataset and model can be changed using the dataset and model arguments. Below is an example of training an Alexnet on CIFAR10 with LSPP:<br>
```
python main.py --dataset cifar10 --model alexnet --method lspp
```

Replace cifar10 with the appropriate dataset and alexnet with the appropriate model. <br>
Supported datasets: CIFAR10, CIFAR100, FashionMNIST, SVHN, TinyImageNet, Animals10N, and ImageNet100. <br><br>
CIFAR10, CIFAR100, FashionMNIST, and SVHN datasets will be downloaded to the path specified in the "data_path" argument (default: "./data").<br>
TinyImageNet, Animals10N, and ImageNet100 need to be downloaded, and the path needs to be provided using the "data_path" argument. 

If you find this paper/code helpful, please cite our paper:
```
```
