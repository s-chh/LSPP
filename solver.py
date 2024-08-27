import os
import torch
import torch.nn as nn
from lspp import LSPP
import torch.nn.functional as F
from data_loader import data_loaders
from sklearn.metrics import accuracy_score
from models import models_small, models_big

class Solver(object):
	def __init__(self, args, load=False):
		self.args = args

		# Create data loaders
		self.train_loader, self.test_loader = data_loaders(self.args)																	

		# Create models
		if self.args.image_size < 224:																									
			self.net = getattr(models_small, self.args.model)(n_classes=self.args.n_classes, n_channels=self.args.n_channels).cuda()	
		else:
			self.net = getattr(models_big,   self.args.model)(pretrained=self.args.pretrained, n_classes=self.args.n_classes).cuda()	

		# Training loss function
		if self.args.method == 'lspp':																									
			self.loss_fn = LSPP(self.args.n_classes, self.args.margin).cuda()	# LS++ loss function
		else:
			self.loss_fn = nn.CrossEntropyLoss().cuda()							# Base cross-entropy loss

	def train(self):
		iter_per_epoch = len(self.train_loader)
		print(f"Iters per epoch: {iter_per_epoch}")

		# Define optimizer for training the model
		if self.args.apply_wd == 1:												# Whether to apply weight decay on c-matrix or not
			trainable_params = list(self.net.parameters()) + list(self.loss_fn.parameters())
			optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
		else:
			trainable_params = [{'params': self.net.parameters(),     'weight_decay': self.args.weight_decay},
								{'params': self.loss_fn.parameters(), 'weight_decay': 0}]
			optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr, momentum=self.args.momentum)
		
		# scheduler for linear warmup of lr and then step decay
		linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/self.args.warmup, end_factor=1.0, total_iters=self.args.warmup-1)
		steplr_decay  = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.lr_drop_epochs, gamma=self.args.lr_drop, last_epoch=self.args.warmup)

		# Variable to capture best test accuracy
		best_acc = 0

		# Training loop
		for epoch in range(self.args.epochs):
			self.net.train()

			new_lr = optimizer.param_groups[0]['lr']
			print(f"\nEp:[{epoch + 1}/{self.args.epochs}]\tlr:{new_lr:.4f}")

			# Loop on loader
			for i, (x, y) in enumerate(self.train_loader):
				x, y = x.cuda(), y.cuda()

				logits = self.net(x)							# Get output logits from the model 
				loss = self.loss_fn(logits, y)					# Compute training los

				# Updating the model
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Log training progress
				if i % 50 == 0:
					print(f'It: {i + 1}/{iter_per_epoch}\tloss:{loss.item():.4f}')

			# Test and save model
			test_acc = self.test(train=(epoch+1)%25==0)               # Test training set every 25 epochs
			if test_acc >= best_acc:
				best_acc = test_acc
				torch.save(self.net.state_dict(), os.path.join(self.args.model_path, self.args.model_name))
				if self.args.c_matrix:
					torch.save(self.loss_fn.state_dict(), os.path.join(self.args.model_path, "c_matrix.pt"))
			print(f"Best test acc: {best_acc:.2%}")
			
			if self.args.c_matrix and self.args.cm:
				self.display_c_matrix()

			# Update learning rate using schedulers
			if epoch < self.args.warmup:
				linear_warmup.step()
			else:
				steplr_decay.step()

	def test_dataset(self, loader):
		# Set model to evaluation mode
		self.net.eval()

		# Arrays to record all labels and logits
		all_labels = []
		all_preds  = []

		# Testing loop
		for (x, y) in loader:
			x = x.cuda()

			# Avoid capturing gradients in evaluation time for faster speed
			with torch.no_grad():
				logits = self.net(x)
				pred = logits.max(1)[1]

			all_labels.append(y)
			all_preds.append(pred.cpu())

		# Convert all captured variables to torch
		all_labels = torch.cat(all_labels)
		all_preds  = torch.cat(all_preds)
		
		# Compute accuracy
		acc  = accuracy_score(y_true=all_labels, y_pred=all_preds)

		return acc

	def test(self, train=False):
		if train:
			acc = self.test_dataset(self.train_loader)
			print(f"Train Accuracy: {acc:.2%}")

		acc = self.test_dataset(self.test_loader)
		print(f"Test Accuracy: {acc:.2%}")
		return acc

	# Display learned C-Matrix
	def display_c_matrix(self):
		# Convert logits of c_matrix to probs
		c_matrix = F.softmax(self.loss_fn.c_matrix, 1)                  			# K, K-1

		# Add 0 at y index to get K X K C-Matrix
		c_matrix = c_matrix.reshape(-1, c_matrix.shape[1]+1)            			# K, K-1    ->  K-1, K       
		c_matrix = F.pad(c_matrix, (1, 0, 0, 0))                        			# K-1, K    ->  K-1, K+1     Pad with 0 after K values starting at index 0
		c_matrix = c_matrix.reshape(-1)                                 			# K-1, K+1  ->  K^2 - 1
		c_matrix = F.pad(c_matrix, (0, 1))                              			# K^2 - 1   ->  K^2          Add 0 at the end
		c_matrix = c_matrix.reshape(self.args.n_classes, self.args.n_classes)       # K^2       ->  K, K

		c_matrix = c_matrix.detach().cpu().numpy()
		print("\nLearned C-Matrix:")
		for row in c_matrix:
			for val in row:
				print(f"{val:.2f} ", end='')
			print()
		print()
