import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, n_classes=10, n_channels=3):
		super(ResNet, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.fc = nn.Linear(512*block.expansion, n_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, deep=False):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		# out = F.avg_pool2d(out, 4)
		out = F.adaptive_avg_pool2d(out, 1)
		x_deep = out.view(out.size(0), -1)
		out = self.fc(x_deep)
		if deep:
			return x_deep, out
		return out


def resnet18(**kwargs):
	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	return model


def resnet34(**kwargs):
	model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	return model


def resnet50(**kwargs):
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	return model


def resnet101(**kwargs):
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	return model


def resnet110(**kwargs):
	model = ResNet(Bottleneck, [3, 4, 26, 3], **kwargs)
	return model


def resnet152(**kwargs):
	model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	return model



class AlexNet(nn.Module):

	def __init__(self, n_classes=10, n_channels=3):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(n_channels, 64, kernel_size=11, stride=4, padding=5),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.classifier = nn.Linear(256, n_classes)
		
	def forward(self, x):
		x = self.features(x)
		x_deep = x.view(x.size(0), -1)
		x = self.classifier(x_deep)
		return x


def alexnet(**kwargs):
	model = AlexNet(**kwargs)
	return model


class LeNet(nn.Module):
	def __init__(self, n_channels=1, n_classes=10):
		super().__init__()
		self.conv1 = nn.Conv2d(n_channels, 6, 5)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2)
		self.fc1 = nn.Linear(400, 120)
		self.relu3 = nn.ReLU()
		self.fc2 = nn.Linear(120, 84)
		self.relu4 = nn.ReLU()
		self.fc3 = nn.Linear(84, n_classes)

	def forward(self, x):
		y = self.conv1(x)
		y = self.relu1(y)
		y = self.pool1(y)
		y = self.conv2(y)
		y = self.relu2(y)
		y = self.pool2(y)
		y = y.view(y.shape[0], -1)
		y = self.fc1(y)
		y = self.relu3(y)
		y = self.fc2(y)
		y_deep = self.relu4(y)
		y = self.fc3(y_deep)
		return y
	

def lenet(n_classes=10, n_channels=1):
	return LeNet(n_channels=n_channels, n_classes=n_classes)


def densenet121(n_classes=10, n_channels=3):
	from models.densenet import densenet121 as densenet121_net
	return densenet121_net(n_classes=n_classes, n_channels=n_channels)

def shufflenet(n_classes=10, n_channels=3):
	from models.shufflenet import shufflenetv2
	return shufflenetv2(n_classes=n_classes, n_channels=n_channels)
