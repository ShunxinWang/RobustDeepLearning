from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
from torch import optim
import torch.nn as nn
from torch.optim import optimizer
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
 
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import metrics
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

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

# ResNet
class ResNet(LightningModule):
    def __init__(self, block,  num_blocks,  num_classes=10, lr=0.0002512):
        super(ResNet, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.test_acc = metrics.Accuracy()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        encoding  = out.view(out.size(0), -1)
        out = self.linear(encoding)
        return encoding, out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,weight_decay=5e-04)
        scheduler = ReduceLROnPlateau(optimizer,factor=0.2, mode='min',verbose=True)#, step_size=2, gamma=0.95)
        return {'optimizer': optimizer, 
                'lr_scheduler':scheduler,
                'monitor': 'train_loss'}   #[optimizer],[scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        self.log_dict({'train_loss': loss}, on_epoch=True,on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
        criterion = nn.CrossEntropyLoss()
        self.val_loss = criterion(y_hat, y)
        self.log_dict( {'val_loss':  self.val_loss}, on_epoch=True,on_step=True)
        return  self.val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self(x)
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


def main():
    logger = TensorBoardLogger("lightning", name="ResNet_EN",log_graph=True)

    model = ResNet(BasicBlock, [2, 2, 2, 2])
    early_stopping = EarlyStopping('val_loss',patience=30)
    trainer = pl.Trainer(progress_bar_refresh_rate=0,logger=logger,gpus= -1, max_epochs=100, callbacks=early_stopping)#, gpus='0',log_every_n_steps=10,val_check_interval=0.25)
   
    trainer.fit(model)
    trainer.test()

if __name__ == '__main__':
    main()