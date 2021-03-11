import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import optimizer
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE as tsne

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import metrics
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Upconvblock(nn.Module):
    expansion = 16
    def __init__(self,in_planes,output_channels, stride=1):
        super(Upconvblock,self).__init__()
        if stride == 1:
            self.scaleup = nn.Conv2d(in_planes,output_channels,kernel_size=3,padding=1)
        elif stride == 2:
            self.scaleup = nn.ConvTranspose2d(in_planes,output_channels,kernel_size=2,stride=stride)
        # self.scaleup = nn.ConvTranspose2d(in_planes,output_channels,kernel_size=2,stride=2)
        # self.scaleup = nn.Upsample(scale_factor=stride,mode="bilinear",align_corners=True)
        self.conv1 = nn.Conv2d(output_channels,output_channels,stride = 1,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels,output_channels,stride=1,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
    
    def forward(self, x):
        x = self.scaleup(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        return x

class ResNet_AE(LightningModule):
    def __init__(self, block_en, block_de, num_blocks, latent_dims=256, lr = 0.000437):
        super(ResNet_AE, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.test_acc = metrics.Accuracy()

        self.in_planes = 64
        self.latent_dims = latent_dims
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block_en, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_en, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_en, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_en, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block_en.expansion, latent_dims)
        self.linear2 = nn.Linear(latent_dims,512*block_de.expansion)
        # self.layer5 = self._make_layer(block_de,512,num_blocks[3],stride=2)
        self.layer6 = self._make_uplayer(block_de,256,num_blocks[2],stride=2)
        self.layer7 = self._make_uplayer(block_de,128,num_blocks[1], stride=2)
        self.layer8 = self._make_uplayer(block_de,64,num_blocks[0],stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,3,kernel_size=3,padding=1),
            nn.Sigmoid()
        )
        self.classifer = nn.Sequential(
            nn.Linear(self.latent_dims,64),
            nn.Linear(64,32),
            nn.Linear(32,10)
        )
        


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def _make_uplayer(self, block,planes,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1) # flatten
        enc = self.linear(out)
        
        prediction = self.classifer(enc)

        out = self.linear2(enc)
        out = out.view(-1,512,4,4)
        # out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.conv2(out)
        return enc, out, prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=5e-04)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',verbose=True, factor=0.2)#, step_size=2, gamma=0.95)
        return {'optimizer': optimizer, 
                'lr_scheduler':scheduler,
                'monitor': 'train_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        _,rct, y_hat = self(x)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.BCELoss()
        loss1 = criterion1(y_hat, y)
        loss2 = criterion2(rct,x)
        loss = loss1 + loss2
        self.log_dict({'train_classification_loss': loss1}, on_epoch=True,on_step=True)
        self.log_dict({'train_reconstruction_loss': loss2}, on_epoch=True,on_step=True)
        self.log_dict({'train_loss': loss}, on_epoch=True,on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _,rct, y_hat = self(x)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.BCELoss()
        loss1 = criterion1(y_hat, y)
        loss2 = criterion2(rct,x)
        self.val_loss = loss1 + loss2
        self.log_dict( {'val_loss':  self.val_loss}, on_epoch=True,on_step=True)
        return  self.val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        _,_, y_hat = self(x)
        _, predicted = torch.max(y_hat.data,1)
        self.test_acc(predicted, y)
        self.log_dict( {'test_acc': self.test_acc}, on_epoch=True,on_step=False)
      
    def setup(self, stage):
        # transform
        transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()
        ])
        transform=transforms.Compose([transforms.ToTensor()])
        cifar_train =torchvision.datasets.CIFAR10(root='./datasets',train=True, download=False, transform=transform_train)
        cifar_test = torchvision.datasets.CIFAR10(root='./datasets',train=False, download=False, transform=transform)

        # train/val split
        cifar_train2, cifar_val =  torch.utils.data.random_split(cifar_train, [45000, 5000])

        # assign to use in dataloaders
        self.train_dataset = cifar_train2
        self.val_dataset = cifar_val
        self.test_dataset = cifar_test

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=100, shuffle=True,num_workers=64)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=100, shuffle=False,num_workers=64)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=100,num_workers=64)

def ResNet18AE(latent_dims = 64):
    return ResNet_AE(BasicBlock, Upconvblock, [2, 2, 2, 2],latent_dims=latent_dims)


def main():#:args):
    logger = TensorBoardLogger("lightning", name="ResNet_AE",log_graph=True)

    latent_dims = 64
    model = ResNet18AE(latent_dims)
    early_stopping = EarlyStopping('val_loss',patience=30)
    trainer = pl.Trainer(progress_bar_refresh_rate=0,logger=logger,callbacks=early_stopping, gpus= -1,  max_epochs=100)# , gpus='0',log_every_n_steps=10,val_check_interval=0.25)
    trainer.fit(model)
    trainer.test()

if __name__ == '__main__':
    main()