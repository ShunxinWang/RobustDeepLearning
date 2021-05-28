import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision
import torchvision.transforms as transforms

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from lightening_Models.ResNet_AE_L1 import ResNet_AE_L1
from lightening_Models.ResNet_AE_L2 import ResNet_AE_L2
from lightening_Models.ResNet_AE_L3 import ResNet_AE_L3
from lightening_Models.ResNet_AE_L4 import ResNet_AE_L4

from lightening_Models.ResNet_EN_L1 import ResNet_EN_L1
from lightening_Models.ResNet_EN_L2 import ResNet_EN_L2
from lightening_Models.ResNet_EN_L3 import ResNet_EN_L3
from lightening_Models.ResNet_EN_L4 import ResNet_EN_L4
from lightening_Models.Blocks import Upconvblock,BasicBlock

import re

ResNet_models = {'AE_L1': ResNet_AE_L1, 'AE_L2': ResNet_AE_L2, 'AE_L3': ResNet_AE_L3, 'AE_L4': ResNet_AE_L4, 
                'EN_L1':ResNet_EN_L1, 'EN_L2':ResNet_EN_L2, 'EN_L3':ResNet_EN_L3, 'EN_L4':ResNet_EN_L4}

def ResNet_AE(image_size, save_log, depth,weight_alpha, weight_beta,num_class=200, num_classifiers=1,weight_mode='None',shortcut=True):
    logger = TensorBoardLogger(save_log, name=depth,version = 'weight' + str(weight_alpha)+'_'+str(weight_beta))
    pattern = re.compile(r"AE_L"+'[1-4]')
    model_type = pattern.search(depth).group() 
    model = ResNet_models[model_type](BasicBlock, Upconvblock, [2, 2, 2, 2],weight_mode=weight_mode, weight_alpha=weight_alpha,
                                 weight_beta=weight_beta, image_size = image_size,num_class=num_class,num_classifiers=num_classifiers)
    return model,logger

def ResNet_EN(image_size, save_log, depth, num_class=200, num_classifiers=1,shortcut=True):
    logger = TensorBoardLogger(save_log, name=depth,version = 'weight1')
    pattern = re.compile(r"EN_L"+'[1-4]') #\.[^\.]+$")
    model_type = pattern.search(depth).group()
    model = ResNet_models[model_type](BasicBlock, [2, 2, 2, 2], image_size = image_size,num_class=num_class,num_classifiers=num_classifiers,shortcut=shortcut)
    return model,logger

def main(args):
    # naming convention of the trained models
    # Results/ + 'model'+'latent space'+ 'image size' + 'num_classifiers' + 'shortcut(False,True)' + 'weight_mode' / 'weight_alpha'+'weight_beta'
    save_dir = 'Results/'
    
    model_depth = args.model_depth
    pattern = re.compile(r"AE|EN") 
    model_type = pattern.search(model_depth).group() 
    maxepoch = 400
    image_size = args.image_size
    num_class = args.num_class
    num_classifiers = args.num_classifiers
    shortcut = args.shortcut
    

    if model_type == 'AE':
        weight_mode = args.weight_mode
        weight_alpha = args.weight_alpha
        weight_beta = args.weight_beta
        save_file = model_depth + '_' + str(image_size) + '_' + str(num_classifiers)+ '_' + str(shortcut)+ '_' + weight_mode
        print(str(shortcut))
        model,logger = ResNet_AE(image_size=image_size, save_log=save_dir, depth = save_file,
                                 num_class=num_class,weight_mode = weight_mode,weight_alpha=weight_alpha,weight_beta=weight_beta,
                                 num_classifiers=num_classifiers,shortcut=shortcut)
        early_stopping = EarlyStopping('val_loss',patience=30)
        trainer = pl.Trainer(progress_bar_refresh_rate=0,logger=logger,callbacks=early_stopping, gpus= -1,  max_epochs=maxepoch)
        trainer.fit(model)
        trainer.test()
    else: 
        save_file = model_depth + '_' + str(image_size) + '_' + str(num_classifiers)+ '_' + str(shortcut)

        model,logger = ResNet_EN(image_size=image_size,save_log=save_dir, depth = save_file, shortcut=shortcut,num_class=num_class,num_classifiers=num_classifiers)
        early_stopping = EarlyStopping('val_loss',patience=30)
        trainer = pl.Trainer(progress_bar_refresh_rate=0,logger=logger,callbacks=early_stopping, gpus= -1,  max_epochs=maxepoch)
        trainer.fit(model)
        trainer.test()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write parameters')
    parser.add_argument('--model_depth', type=str, default='AE_L4',
                    help='model and latent space')
    parser.add_argument('--image_size', type=int, default= 64,
                    help='size of images in dataset')
    parser.add_argument('--num_class', type=int, default= 200,
                    help='number of classes in dataset')
    parser.add_argument('--num_classifiers', type=int, default=1,
                    help='number of classifiers used')
    parser.add_argument('--shortcut', default=True,
                    help='shortcut removal')
    parser.add_argument('--weight_mode', type=str, default='None',
                    help='None, fixed_schedule_alpha,fixed_schedule_beta,dynamic_alpha,dynamic_beta')
    parser.add_argument('--weight_alpha', type=int, default=1,
                    help='weight of classification loss')
    parser.add_argument('--weight_beta', type=int, default=1,
                    help='weight of reconstruction loss')
    args = parser.parse_args()
    main(args)