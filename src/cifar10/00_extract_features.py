# https://github.com/huyvnphan/PyTorch_CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pickle as pkl
import sys
sys.path.append("./../../PyTorch_CIFAR10")
from cifar10_models import *
import torch.nn.functional as F
import paths

sanity_check_preprocessing = False

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
embedding_size =1664

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean, std)])

complete_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
num_train = int(len(complete_trainset)*.2)
num_empty = int(len(complete_trainset)*.7)
num_test = len(complete_trainset)  - num_train -num_empty
torch.manual_seed(0);

trainset, valset,_= torch.utils.data.random_split(complete_trainset, [num_train, num_test,num_empty])


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


my_model = densenet169(pretrained=True) # use densenet to extract features

if sanity_check_preprocessing: #for sanity check
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = my_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
          
    assert correct / total > 0.5

embeddings = {}
true_y = {}

for name, partition in zip(["val", "train", 'test'],[ valset,trainset,testset ]):
    x = np.empty((len(partition), embedding_size))
    y= np.empty((len(partition)), dtype =np.int16)
    data_loader = torch.utils.data.DataLoader(partition, batch_size=256,
                                         shuffle=False, num_workers=5)
    with torch.no_grad():
        for i, data in enumerate(data_loader,):

            inputs, labels = data
            features = my_model.features(inputs)
            outputs = F.adaptive_avg_pool2d(features , (1, 1)).view(features.size(0), -1)
            y[i*256:(i+1)*256] = labels.numpy()
            x[i*256:(i+1)*256] = outputs.detach().numpy()
    print("Embedded ", name)
    embeddings[name] = x
    true_y[name] = y
pkl.dump( embeddings, open(paths.paths['cifar10_embeddings'] , "wb" ) )
pkl.dump( true_y, open(paths.paths['cifar10_labels'] , "wb" ) )    
    