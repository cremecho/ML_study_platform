import random
import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd
import argparse
import os
import time
from datetime import datetime
import json

from utils.model_setting_helper import complete_settings
from utils import metrics
from utils.model_setting_helper import *
from utils.get_loader import get_loader

class Tester(object):
    def __init__(self, configs):
        # 1. set configs
        if not len(configs) == 9:
            complete_settings(configs)
        for k,v in configs.items():
            setattr(self, k, v)
        # test's save folder is the folder that stores the pth model
        self.save_path = os.path.split(self.pth_path)[0]

        # 2. load model
        self.model = get_model(self)
        checkpoint = torch.load(self.pth_path)
        self.model.load_state_dict(checkpoint)
        self.model.cuda()
        self.model.eval()

        # 3. get dataloader
        self.test_loader = get_loader(self.model_name, self.scale, self.dataset_root, self.batch_size, self.val,
                                                            pin_memory=True, num_workers=1, mode='test')
        # 4. set loss function
        self.criterion = get_criterion(self)
        # 5. set optimizer
        self.optimizer = get_optimizer(self)
        # 6. set metrics
        self.metrics = metrics.Metrics(self.save_path, self.model_name)


    def testing(self):
        epoch_loss = 0.
        for iteration, batch in enumerate(self.test_loader):
            x, y = batch['x'], batch['y']
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                output = self.model(x)
            output = output.squeeze()
            loss = self.criterion(output, y.float())
            epoch_loss += loss.item()
            self.metrics.add_batch(y, output)
        epoch_loss /= len(self.test_loader)
        self.logging(epoch_loss, 'test')
        self.metrics.confusion_matrix_map(0, 'test')
        self.metrics.reset()


    def logging(self, epoch_loss, mode):
        str_loss = '%s, loss=%.5f' % (mode, epoch_loss)
        # MEL is class 0
        str_metrics = 'Acc: %.3f, ' \
                      'Class Acc: %.3f, Sensitivity: %.3f, ' \
                      'Specificity: %.3f, F1-Score: %.3f' % (
                          self.metrics.Accuracy(), self.metrics.Accuracy_Class(),
                          self.metrics.Sensitivity(0),
                          self.metrics.Specificity(0), self.metrics.F1(0))
        print(str_loss)
        print(str_metrics)
        with open(os.path.join(self.save_path, mode + '.txt'), 'w') as f:
            print(str_loss, file=f)
            print(str_metrics, file=f)


if __name__ == '__main__':
    with open('./configs/_common_config.json', 'r') as f:
        common_config = json.load(f)
    with open('./configs/' + common_config['model_name'] + '.json', 'r') as f:
        model_config = json.load(f)
    model_config.update(common_config)
    tester = Tester(model_config)
    tester.testing()