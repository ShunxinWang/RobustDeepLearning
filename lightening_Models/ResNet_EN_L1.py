import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import sys
sys.path.insert(0,'/home/wangs1/')
from datasets.TinyImageNet import TinyImageNet
import torchvision.transforms as transforms

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


class ResNet_EN_L1(LightningModule):
    def __init__(self, block_en, num_blocks, lr = 0.001, image_size = 64, num_class=200, num_classifiers = 1,shortcut=True):
        super(ResNet_EN_L1, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.test_acc = metrics.Accuracy()
        self.in_planes = 64
        self.image_size = image_size
        self.num_class = num_class
        self.num_classifiers = num_classifiers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block_en, 64, num_blocks[0], stride=1,shortcut=True)
        self.layer2 = self._make_layer(block_en, 128, num_blocks[1], stride=2,shortcut=shortcut)
        self.layer3 = self._make_layer(block_en, 256, num_blocks[2], stride=2,shortcut=shortcut)
        self.layer4 = self._make_layer(block_en, 512, num_blocks[3], stride=2,shortcut=shortcut)

       
        self.classifier4 = nn.Linear(64,self.num_class)
        if self.num_classifiers == 4:
            self.classifier = nn.Linear(512,self.num_class)
            self.classifier2 = nn.Linear(256,self.num_class)
            self.classifier3 = nn.Linear(128,self.num_class)
        
        


    def _make_layer(self, block, planes, num_blocks, stride,shortcut):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,shortcut))
            self.in_planes = planes 
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        enc = F.avg_pool2d(out, int(self.image_size))
        enc = enc.view(enc.size(0), -1) # flatten
        prediction4 = self.classifier4(enc)
        out = self.layer2(out)
        if self.num_classifiers == 4:
            enc = F.avg_pool2d(out, int(self.image_size/2))
            enc = enc.view(enc.size(0), -1) # flatten
            prediction3 = self.classifier3(enc)
        out = self.layer3(out)
        if self.num_classifiers == 4:
            enc = F.avg_pool2d(out, int(self.image_size/4))
            enc = enc.view(enc.size(0), -1) # flatten
            prediction2 = self.classifier2(enc)
        out = self.layer4(out)
        if self.num_classifiers == 4:
            enc = F.avg_pool2d(out, int(self.image_size/8))
            enc = enc.view(enc.size(0), -1) # flatten
            prediction = self.classifier(enc)
        if self.num_classifiers == 4:
            return enc, prediction, prediction2, prediction3, prediction4
        return enc, prediction4

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=5e-04)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',verbose=True, factor=0.2)#, step_size=2, gamma=0.95)
        return {'optimizer': optimizer, 
                'lr_scheduler':scheduler,
                'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        criterion1 = nn.CrossEntropyLoss()
        if self.num_classifiers == 4:
            _, y_hat, y_hat2, y_hat3, y_hat4 = self(x)
            loss1 = criterion1(y_hat, y) + criterion1(y_hat2, y) + criterion1(y_hat3, y) + criterion1(y_hat4, y) 

        else:
            _, y_hat = self(x)
            loss1 = criterion1(y_hat, y)
        loss = loss1
       
        self.log_dict({'train_classification_loss': loss1}, on_epoch=True,on_step=True)
        self.log_dict({'train_loss': loss}, on_epoch=True,on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        criterion1 = nn.CrossEntropyLoss()
        if self.num_classifiers == 4:
            _, y_hat, y_hat2, y_hat3, y_hat4 = self(x)
            loss1 = criterion1(y_hat, y) + criterion1(y_hat2, y) + criterion1(y_hat3, y) + criterion1(y_hat4, y) 

        else:
            _, y_hat = self(x)
            loss1 = criterion1(y_hat, y)
        self.val_loss = loss1
        self.log_dict( {'val_loss':  self.val_loss}, on_epoch=True,on_step=True)
        return  self.val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.num_classifiers == 4:
            _, y_hat, y_hat2, y_hat3, y_hat4 = self(x)
            all_yhat = torch.cat((y_hat.data,y_hat2.data,y_hat3.data,y_hat4.data),1)
            scores1, predicted1 = torch.max(all_yhat.view(x.size(0),4,self.num_class),2)
            _, predicted2 = torch.max(scores1,1)
            predicted = [int(c[k]) for c,k in zip(predicted1,predicted2)]
            predicted = torch.IntTensor(predicted).to(device='cuda:0')

        else:
            _, y_hat = self(x)
            _, predicted = torch.max(y_hat.data,1)
        self.test_acc(predicted, y)
        self.log_dict( {'test_acc': self.test_acc}, on_epoch=True,on_step=False)
      
    def setup(self, stage):
        # transform
        transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.image_size),
            transforms.ToTensor()
        ])
        transform=transforms.Compose([transforms.ToTensor()])

        if self.image_size == 64:
            print('Tiny')
            data_train =TinyImageNet(root='./datasets',train=True,  transform=transform_train)
            data_test = TinyImageNet(root='./datasets',train=False, transform=transform)

        elif self.image_size == 32:
            data_train = torchvision.datasets.CIFAR10(root='./datasets',train=True, download=False, transform=transform_train)
            data_test = torchvision.datasets.CIFAR10(root='./datasets',train=False, download=False, transform=transform)

        # train/val split
        data_train2, data_val =  torch.utils.data.random_split(data_train, [int(len(data_train)*0.9), len(data_train)-int(len(data_train)*0.9)])

        # assign to use in dataloaders
        self.train_dataset = data_train2
        self.val_dataset = data_val
        self.test_dataset = data_test

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=100, shuffle=True,num_workers=64)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=100, shuffle=False,num_workers=64)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=100,num_workers=64)

