# Label Smoothing++ or LS++
Official Implementation of paper "Label Smoothing++: Enhanced Label Regularization for Training Neural Networks". <br>

## Table of Contents
- [Introduction](#introduction)
  - [Overview](#overview)
  - [Algorithm](#algorithm)
- [Usage](#usage)
  - [Requirements](#requirements)
  - [Run Commands](#run-commands)
  - [Data](#data)
  - [PyTorch Code for Label Smoothing++][#quick-pytorch-code-for-label-smoothing++]
- [Results](#results)
- [Cite](#cite)

## Introduction
### Overview
- Label smoothing++ is a label regularization technique that learns optimal training targets.
- Learns a C-Matrix where each class has a unique probability vector for non-target classes.
	- For a **K** class classification task, each class learns a unique **K-1** probability vector for non-target classes.
	- These vectors are combined to form a C matrix (**K x K**) by adding the target class (itself) probability of 0.
- Probability vectors from the C matrix are combined with a one-hot vector to create the final training target.
- The target class probability is fixed but the non-target class probabilities are flexible.
- Key benefits:
    - Encourages more flexible learning.
    - Improves network robustness and accuracy.

### Algorithm
1. **Initialize the C-Matrix** with different learnable probability vectors for each class.
2. For each sample, create its **1-hot vector** of the target class.
3. Use the **C-Matrix** to fetch the corresponding **probability vector** for non-target classes.
4. **Combine** the 1-hot vector and the C-Matrix vector using a **weighted sum** controlled by a **hyperparameter** 𝛼 α.
5. Optimize the network parameters with **cross-entropy loss**.
6. Train the C-Matrix with the **reverse cross-entropy loss**.
7. Repeat steps 2–6 until convergence.

## Usage
### Requirements
Python, scikit-learn, PyTorch, and torchvision
 
### Run command:
To train a model using Label Smoothing++ (LS++), use **lspp** as the **method** argument. For instance, to train AlexNet on CIFAR10:
```
python main.py --dataset cifar10 --model alexnet --method lspp
```

The `--apply_wd` argument controls whether weight decay should be applied to the C-Matrix. Setting this to `False` results in a sharper C-Matrix but may reduce performance due to overfitting.

### Data
- To change the dataset, **replace CIFAR10** with the appropriate dataset. <br>
- **CIFAR10**, **CIFAR100**, **FashionMNIST**, and **SVHN** are automatically downloaded by the script.
- **TinyImageNet**, **Animals10n**, and **Imagenet100** need to be downloaded manually.
#### Dataset Directory Structure
For manually downloaded datasets, organize the data in the following directory structure:
```
data/
├── train/
│ ├── class1/
│ ├── class2/
├── test/
│ ├── class1/
│ ├── class2/
```
#### Dataset Links
Here are the links to download the required datasets:
- [TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)  
- [Animals10N](https://dm.kaist.ac.kr/datasets/animal-10n/)  
- [ImageNet100](https://www.kaggle.com/datasets/ambityga/imagenet100)  

#### Specifying Dataset Paths
For manually downloaded datasets, use the `--data_path` argument to specify the path to the dataset. Example:
```bash
python main.py --method lspp --model resnet18 --dataset tinyimagenet --data_path /path/to/data
```

### Quick PyTorch Code for Label Smoothing++
Alternatively, simple PyTorch code for quick integration with other frameworks:
```
from lspp import LSPP

# Define loss function
loss_fn = LSPP(num_classes, alpha=0.1).cuda()

# Add C-Matrix to the training parameters of the optimizer
opt = SGD(list(model.parameters()) + list(loss_fn.parameters()), lr, mom, wd)
.
.
# Calculate loss
loss = loss_fn(logits, targets)
opt.zero_grad()
loss.backward()
opt.step()
```

## Cite
If you found our work/code helpful, please cite our paper:
```
Bibtex upcoming
```
