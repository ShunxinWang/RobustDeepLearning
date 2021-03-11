from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import metrics
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

# -----------------------------------------------------------------------------------------
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class Reshape(nn.Module):
    def forward(self, input):
        return input.view(-1,512,8,8)

class Regular_En(LightningModule):
    def __init__(self,laten_dims: int,lr=0.00109648):
        super(Regular_En, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.laten_dims = laten_dims
        self.test_acc = metrics.Accuracy()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, 4, stride=2, padding=1), 
            nn.BatchNorm2d(128),           
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),     
            nn.BatchNorm2d(256),     
            nn.ReLU(),
			nn.Conv2d(256, 512, 4, stride=2, padding=1),  
            nn.BatchNorm2d(512),         
            nn.ReLU(),
            Flatten(),
            nn.Linear(4*4*512,self.laten_dims)
# 			
        )
        
        self.classifer = nn.Sequential(
            nn.Linear(self.laten_dims,64),
            nn.Linear(64,32),
            nn.Linear(32,10)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        prediction = self.classifer(encoded)
        return encoded, prediction
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-04)
        scheduler = ReduceLROnPlateau(optimizer,factor=0.2, mode='min',verbose=True)
        return {'optimizer': optimizer, 
                'lr_scheduler':scheduler,
                'monitor': 'train_loss'}


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
        self.log_dict( {'test_acc': self.test_acc},  on_epoch=True,on_step=False)
        

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


def main():#:args):
    logger = TensorBoardLogger("lightning", name="Regular_EN",log_graph=True)

    latent_dims = 64
    model = Regular_En(latent_dims)
    early_stopping = EarlyStopping('val_loss', patience=30)
    trainer = pl.Trainer(progress_bar_refresh_rate=0,logger=logger,gpus= -1, callbacks=early_stopping, max_epochs=200)#, gpus='0',log_every_n_steps=10,val_check_interval=0.25)
   
    trainer.fit(model)
    trainer.test()

if __name__ == '__main__':
    main()