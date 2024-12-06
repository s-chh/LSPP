# Label Smoothing++ or LS++
Official Implementation of paper "Label Smoothing++: Enhanced Label Regularization for Training Neural Networks". <br>

## Introduction: (To be fixed)
Label Smoothing++ enables neural networks to learn separate targets for each individual class. It ensures that samples within the same class yield consistent outputs.
select their optimal training labels. It uses different training labels for each class while ensuring that samples within the same class yield consistent outputs. The class-wise probability vector to add to the 1-hot vector of each class. The learned targets regularize the network and provide improved performance.
designed to <br>

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
