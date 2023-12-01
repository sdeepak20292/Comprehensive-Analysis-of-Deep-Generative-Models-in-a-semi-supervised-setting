import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, STL10
import os
import pickle
import zipfile
import datetime
import torch.utils.data as tud
from m2_vae import M2,Classifier
from baseline_cnn import BaselineConvNet

data_transform = transforms.Compose([
                transforms.ToTensor()
        ])
train = STL10(root="./data", split="train", transform=data_transform, download=True)
test = STL10(root="./data", split="test", transform=data_transform, download=True)


train_loader = torch.utils.data.DataLoader(test, batch_size=8000, shuffle=False, num_workers=0)
data, labels= next(iter(train_loader))
np.random.seed(5)
labeled_ind = np.random.choice(8000,1500, replace = False)

unlabeled_ind = np.setdiff1d(list(range(8000)), labeled_ind)
labels = labels.numpy()
np.put(labels,list(unlabeled_ind),10)
np.random.seed(5)
dev_ind = labeled_ind[np.random.choice(1500,450, replace = False)]
train_ind = np.setdiff1d(list(range(8000)), dev_ind)

#prepare dataloader for pytorch
class TorchInputData(tud.Dataset):
    """
    A simple inheretance of torch.DataSet to enable using our customized DogBreed dataset in torch
    """
    def __init__(self, X, Y, transform=None):
        """
        X: a list of numpy images 
        Y: a list of labels coded using 0-9 
        """        
        self.X = X
        self.Y = Y 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        return x, y
    

images_train = [data[i] for i in train_ind]
trainset = TorchInputData(images_train, labels[train_ind])
train_loader = tud.DataLoader(trainset, batch_size=50, shuffle=True)

images_dev = [data[i] for i in dev_ind]
devset = TorchInputData(images_dev, labels[dev_ind])
dev_loader = tud.DataLoader(devset, batch_size=50, shuffle=True)


classifier = Classifier(image_reso = 96, filter_size = 5, dropout_rate = 0.2)
m2 = M2(latent_features = 128, classifier = classifier, path = "m2_stl10_0.1_50epoch_5.pth")
alpha = 0.1*len(train_loader.dataset)
m2.fit(train_loader,dev_loader,50,alpha,labeled_data_len = 1500)

## baseline model(CNN)
dev_ind_b = dev_ind
#training data is the same 1050 labeled data as M2
train_ind_b = (np.setdiff1d(labeled_ind, dev_ind))

images_train_b = [data[i] for i in train_ind_b]
trainset_b = TorchInputData(images_train_b, labels[train_ind_b])
train_loader_b = tud.DataLoader(trainset_b, batch_size=50, shuffle=True)
images_dev_b = [data[i] for i in dev_ind_b]
devset_b = TorchInputData(images_dev_b, labels[dev_ind_b])
dev_loader_b = tud.DataLoader(devset_b, batch_size=50, shuffle=True)
baseline = BaselineConvNet(96, path = "baseline_stl10_100epoch_5.pth")
print(baseline.model)
baseline.fit(train_loader_b,dev_loader_b)
baseline.train(100)





## testing on test set.
testset_loader = torch.utils.data.DataLoader(train, batch_size=1000, shuffle=True, num_workers=0)
# baseline cnn model
conf_b, acc_b = baseline.test(testset_loader,path = "baseline_stl10_100epoch_5.pth",return_confusion_matrix = True)
#m2 model
conf, acc = m2.test(testset_loader,path = "m2_stl10_0.1_50epoch_5.pth",return_confusion_matrix = True)
