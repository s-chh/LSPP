# Label Smoothing++ or LS++
Official Implementation of paper "Label Smoothing++: Enhanced Label Regularization for Training Neural Networks". <br>

## Introduction
**1-line summary:** A label regularization method that learns the optimal training probabilities for non-target classes, tailored to each class.

### Overview
- Label Smoothing++ allows the network to learn optimal probability assignment.
- Each class learns a different probability assignment for all the non-target classes.
- The target class probability is fixed but the non-target class probabilities are flexible.
- Combined probability vectors form a C-Matrix by setting the diagonal (itself) to 0.
  
### Algorithm
1. Initialize the C-Matrix with different learnable probability vectors for each class.
2. For each sample, create its 1-hot vector of the target class.
3. Get the associated probabilities vector from the C-Matrix.
4. Combine the two probability vectors using a weighted sum (α is the weight).
5. Train the network using cross-entropy loss only (using stop gradients).
6. Train the C-Matrix using the reverse cross-entropy loss only (using stop gradients).
7. Repeat 2-6

## Usage
### Requirements
- Python
- scikit-learn
- PyTorch
- torchvision
 
### Run command:
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

### Data
- To change the dataset, **replace cifar10** with the **appropriate dataset**. <br>
- Cifar10, Cifar100, FashionMNIST, and SVHN will be auto-downloaded,
	-  At the path specified in the "data_path" argument (default: "./data").
- TinyImageNet, Animals10n, and Imagenet100 need to be downloaded.
   - Data must be split into 'train' and 'test' folders. 
   - Path needs to be provided using the "data_path" argument.
- Dataset links:
   - TinyImageNet: <a href="http://cs231n.stanford.edu/tiny-imagenet-200.zip">http://cs231n.stanford.edu/tiny-imagenet-200.zip</a> 
   - Animals10N: <a href="https://dm.kaist.ac.kr/datasets/animal-10n/">https://dm.kaist.ac.kr/datasets/animal-10n/</a>  
   - ImageNet100: <a href="https://www.kaggle.com/datasets/ambityga/imagenet100">https://www.kaggle.com/datasets/ambityga/imagenet100/</a>  

#### Simple PyTorch Code for Label Smoothing++
Alternatively, simple PyTorch code for integrating with other frameworks:
```
# Define LS++ Class
class LSPP(nn.Module):
	def __init__(self, K, alpha=0.1):
		super().__init__()
		self.K = K
		self.alpha = alpha
		self.c_matrix = nn.Parameter(torch.zeros(K, K-1), requires_grad=True)

	def forward(self, logits, y):
		pred = F.softmax(logits, 1)

		y_1hot = F.one_hot(y, num_classes=self.K).float()

		# Convert logits of c_matrix to probs
		c_matrix = F.softmax(self.c_matrix, 1)        # K, K-1

		# Add 0 at y indices to get the C-Matrix of size K x K
		c_matrix = c_matrix.reshape(-1, self.K)       # K, K-1   -> K-1, K       
		c_matrix = F.pad(c_matrix, (1, 0, 0, 0))      # K-1, K   -> K-1, K+1     
		c_matrix = c_matrix.reshape(-1)               # K-1, K+1 -> K^2 - 1
		c_matrix = F.pad(c_matrix, (0, 1))            # K^2 - 1  ->  K^2        
		c_matrix = c_matrix.reshape(self.K, self.K)   # K^2      ->  K, K

		# Compute Targets
		y_tgt = (1 - self.alpha) * y_1hot + self.alpha * c_matrix[y]

		# Symmetric cross-entropy loss with detach
		fwd_ce = cross_entropy_loss(y_tgt, pred.detach())
		bck_ce = cross_entropy_loss(pred, y_tgt.detach())
		loss = (fwd_ce + bck_ce) / 2
		return loss

# Define loss function
loss_fn = LSPP(K, α)

# Add C-Matrix to the training parameters
opt = SGD(list(net.parameters()) + list(loss_fn.parameters()), lr, mom, wd)
```

## Cite
If you found our work/code helpful, please cite our paper:
```
Bibtex upcoming
```
