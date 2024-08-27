import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_loss(y_hat, y_target):
	loss = - y_target * (y_hat + 1e-6).log()
	loss = loss.sum(1).mean()
	return loss


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
		c_matrix = F.softmax(self.c_matrix, 1)               # K, K-1

		# Add 0 at y indices to get K X K C-Matrix
		c_matrix = c_matrix.reshape(-1, self.K)            	 # K, K-1    ->  K-1, K       
		c_matrix = F.pad(c_matrix, (1, 0, 0, 0))             # K-1, K    ->  K-1, K+1     Pad with 0 after K values starting at index 0
		c_matrix = c_matrix.reshape(-1)                      # K-1, K+1  ->  K^2 - 1
		c_matrix = F.pad(c_matrix, (0, 1))                   # K^2 - 1   ->  K^2          Add 0 at the end
		c_matrix = c_matrix.reshape(self.K, self.K)          # K^2       ->  K, K

		# Compute Targets
		y_tgt = (1 - self.alpha) * y_1hot + self.alpha * c_matrix[y]

		# Symmetric cross-entropy loss with detach
		fwd_ce = cross_entropy_loss(y_tgt, pred.detach())
		bck_ce = cross_entropy_loss(pred, y_tgt.detach())
		loss = (fwd_ce + bck_ce) / 2
		return loss
