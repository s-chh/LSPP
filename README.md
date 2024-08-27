# Label Smoothing++ (LS++)
This is the official PyTorch Implementation of our upcoming BMVC 2024 PatchRot paper "Label Smoothing++: Enhanced Label Regularization for Training Neural Networks". <br>

## Introduction
PatchRot rotates images and image patches and trains the network to predict the rotation angles. 
The network learns to extract global image and patch-level features through this process. 
PatchRot pretraining extracts superior features and provides improved performance. <br>

## Run commands (also available in <a href="scripts.sh">scripts.sh</a>):
Run <strong>main_pretrain.py</strong> to pre-train the network with PatchRot, followed by <strong>main_finetune.py --init patchrot</strong> to finetune the network.<br>
<strong>main_finetune.py --init none</strong> can be used to train the network without any pretraining (training from random initialization).<br>
Below is an example on CIFAR10:

| Method | Run Command |
| :---         | :---         |
| PatchRot pretraining | python main_pretrain.py --dataset cifar10 |
| Finetuning pretrained model | python main_finetune.py --dataset cifar10 --init patchrot |
| Training from random init | python main_finetune.py --dataset cifar10 --init none |

Replace cifar10 with the appropriate dataset. <br>
Supported datasets: CIFAR10, CIFAR100, FashionMNIST, SVHN, TinyImageNet, Animals10N, and ImageNet100. <br><br>
CIFAR10, CIFAR100, FashionMNIST, and SVHN datasets will be downloaded to the path specified in the "data_path" argument (default: "./data").<br>
TinyImageNet, Animals10N, and ImageNet100 need to be downloaded, and the path needs to be provided using the "data_path" argument. 

If you find this paper/code helpful, please cite our paper:
```
```
