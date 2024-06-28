import random
import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd
import os
import time
from datetime import datetime
import json

from utils import metrics
from utils.model_setting_helper import *
from utils.get_loader import get_loader
from utils.ploting import *

def save_config(config):
    """
    make saving dir, format: save_path/YYYY-MM-DD HH-MM-SS
    clear empty folder in save_path
    """

    # clear empty folder [python 删除空文件夹（包括各级空子文件夹）代码_python删除空文件夹-CSDN博客](https://blog.csdn.net/zaibeijixing/article/details/135426732)
    def remove_empty_folders(path):
        if not os.path.isdir(path):
            return

        # 遍历当前文件夹
        for root, dirs, files in os.walk(path, topdown=False):
            for name in dirs:
                folder_path = os.path.join(root, name)

                # 如果文件夹为空，则删除
                if not os.listdir(folder_path):
                    os.rmdir(folder_path)
                    print(f"Deleted empty folder: {folder_path}")


    remove_empty_folders(config['save_path'])

    # make save dir
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    second_path = os.path.join(config['save_path'], config['model_name'])
    if not os.path.exists(second_path):
        os.mkdir(second_path)
    dt = datetime.now()
    folder_name = str(dt).replace(':', '-').split('.')[0]
    os.mkdir(os.path.join(second_path, folder_name))
    config['save_path'] = os.path.join(second_path, folder_name)




class Trainer(object):
    def __init__(self, configs):
        """
        initialize all the settings for training
        including: loading model; get dataloader; set loss function;
                   set optimizer; set metrics
        :param configs:
            configs of models as dict, must include:
                "model_name": str, model name
                "save_path": str, path for saving results
                "dataset_root": str, path of dataset
            should include:
                "batch_size": int, batch size, 32 by default
                "epoch": int, epochs to train, 50 by default
                "lr": float, learning rate, 1e-2 by default
                "val": bool, whether validating in training or not, True by default
                "scale": float, [0.0-1.0], 1 for using the hole training set, 1 by default
                "pth_path": str, for testing, absolute path of .pth model to load
        """

        # 1. set configs
        if not len(configs) == 9:
            complete_settings(configs)
        for k,v in configs.items():
            setattr(self, k, v)

        # 2. load model
        model = get_model(self)
        self.model = model.cuda()
        # 3. get dataloader
        if self.val:
            self.train_loader, self.val_loader = get_loader(self.model_name, self.scale, self.dataset_root, self.batch_size, self.val,
                                                            pin_memory=True, num_workers=1, mode='train')
        else:
            self.train_loader = get_loader(self.model_name, self.scale, self.dataset_root, self.batch_size, self.val,
                                           pin_memory=True, num_workers=1, mode='train')

        # 4. set loss function
        self.criterion = get_criterion(self)
        # 5. set optimizer
        self.optimizer = get_optimizer(self)
        # 6. set metrics
        self.metrics = metrics.Metrics(self.save_path, self.model_name)
        # store best validation accuracy
        self.best_pred = 0.


    def train(self):
        for epoch in range(1, self.epoch + 1):
            epoch_loss = 0.
            for iteration, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x,y = batch['x'], batch['y']
                x,y = x.cuda(), y.cuda()
                x = Variable(x)

                output = self.model(x)
                output = output.squeeze()
                loss = self.criterion(output, y.float())
                self.metrics.add_batch(y, output)
                epoch_loss += loss.item()

                # TODO: for larger dataset, add 100 batch output
                loss.backward()
                self.optimizer.step()
            # TODO: make dir, save confusion matrix plots
            self.logging(epoch, epoch_loss/len(self.train_loader), 'train')
            self.metrics.confusion_matrix_map(epoch, 'train')
            self.metrics.reset()
            if self.val:
                self.validation(epoch)
            # save last epoch model
            if epoch == self.epoch - 1:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'last_epoch.pth'))



    def validation(self, epoch):
        epoch_loss = 0.
        for iteration, batch in enumerate(self.val_loader):
            x, y = batch['x'], batch['y']
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                output = self.model(x)
            output = output.squeeze()

            loss = self.criterion(output, y.float())
            epoch_loss += loss.item()
            self.metrics.add_batch_binary(y, output)
        epoch_loss /= len(self.val_loader)
        self.logging(epoch, epoch_loss, 'val')
        # save best predict model
        if self.metrics.Accuracy() > self.best_pred:
            self.best_pred = self.metrics.Accuracy()
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_epoch.pth'))
        self.metrics.confusion_matrix_map(epoch, 'val')
        self.metrics.reset()


    def logging(self,epoch ,epoch_loss, mode):
        str_loss = '%s, epoch: %d, loss=%.5f' % (mode,
            epoch, epoch_loss)
        # MEL is class 0
        str_metrics = 'Acc: %.3f, ' \
                      'Class Acc: %.3f, Sensitivity: %.3f, ' \
                      'Specificity: %.3f, F1-Score: %.3f' % (
                          self.metrics.Accuracy(), self.metrics.Accuracy_Class(),
                          self.metrics.Sensitivity(0),
                          self.metrics.Specificity(0), self.metrics.F1(0))
        print(str_loss)
        print(str_metrics)
        with open(os.path.join(self.save_path, mode+'.txt'), 'a+') as f:
            print(str_loss, file=f)
            print(str_metrics, file=f)


if __name__ == '__main__':
    # loading the configs for the specified model
    with open('./configs/_common_config.json', 'r') as f:
        common_config = json.load(f)
    with open('./configs/' + common_config['model_name'] + '.json', 'r') as f:
        model_config = json.load(f)
    model_config.update(common_config)
    # make the dir for saving results
    save_config(model_config)

    trainer = Trainer(model_config)
    trainer.train()
    plot_metrics(configs=model_config, file_name='train.txt', metrics_ls=[])
    plot_metrics(configs=model_config, file_name='val.txt', metrics_ls=[])
    plot_loss(configs=model_config)