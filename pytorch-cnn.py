# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

# model and dataset
from models import VGG
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

# tensorboardX
from tensorboardX import SummaryWriter

from tqdm import tqdm
import os

data_dir = "./data"
epochs = 50
batch_size = 128
num_classes = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

transform_test = transforms.Compose([transforms.ToTensor()])

train_set = CIFAR100(data_dir, train=True, transform=transform_train, download=True)
test_set = CIFAR100(data_dir, train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)


test_iter = iter(test_loader)
images, _ = test_iter.next()
images = images[0]
print("Dataset stats:\n")
print("Number of training images: {}, testing: {}".format(len(train_set), len(test_set)))
print("Image size: {}".format(images.size()))

print("Loading model...")

model = VGG('VGG19', num_classes)
model.to(device)

if device == 'cuda':
    model = nn.DataParallel(model)
    cudnn.benchmark = True

# define optimizer and loss
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())

# Training phase
model.train()
print_step = len(train_loader) // 50
train_loss = 0
correct = 0
total = 0
print("Training Starting...")
for e in range(epochs):
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        opt.zero_grad()

        # forward
        outputs = model(inputs)

        # backward
        loss = criterion(outputs, targets)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if i % print_step == 0:
            print("Epoch [{} / {}], Batch [{} / {}]: loss: {:.4f}, acc: {:.2f}".format(e+1, epochs, i+1, batch_size, train_loss/(i+1), (correct / total) * 100.0))

    print("Epoch [{} / {}]: loss: {:.4f}, acc: {:.2f}".format(e+1, epochs, train_loss/(batch_size), (correct / total) * 100.0))


# Testing Phase
model.eval()
test_loss = 0
correct = 0
total = 0
for i, data in test_loader:
    inputs, targets = data
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    test_loss += loss.item()
    _, predicted = outputs.max(dim=1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()


print("Testing Stats: loss: {:.4f}, acc: {:.2f}".format(test_loss/(test_set), (correct / total) * 100.0))
