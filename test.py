import torch
import os
import time
from datetime import datetime
import json

from utils import metrics
from utils.model_setting_helper import *


class Tester(object):
    def __init__(self, configs):
        # 1. set configs
        complete_settings(configs)
        for k,v in configs.items():
            setattr(self, k, v)
        # test's save folder is the folder that stores the pth model
        self.save_path = os.path.split(self.pth_path)[0]
        # testing mode
        self.mode = 'test'

        # 2.-5. load model, dataset; set criterion; set optimizer
        model, dataset, loss_func, optimizer = get_model_settings(self)
        self.model = model
        checkpoint = torch.load(self.pth_path)
        self.model.load_state_dict(checkpoint)
        self.model = model.cuda()
        self.model.eval()

        self.criterion = loss_func
        self.optimizer = optimizer(model.parameters(), lr=self.lr)

        # 6. get dataloader
        self.test_loader = get_loader(self, dataset, pin_memory=True, num_workers=1)

        # 7. set metrics
        self.metrics = metrics.Metrics(self.model_name, dataset.NUM_CLASSES, dataset.CLASS_LABELS, self.save_path)


    def testing(self):
        epoch_loss = 0.
        for iteration, batch in enumerate(self.test_loader):
            x, y = batch['x'], batch['y']
            x, y = x.cuda(), y.cuda()
            x = x.float()
            with torch.no_grad():
                output = self.model(x)
            output = output.squeeze()
            if self.criterion._get_name() == 'BCELoss':
                y = y.float()
            loss = self.criterion(output, y)
            epoch_loss += loss.item()
            self.metrics.add_batch(y, output)
        epoch_loss /= len(self.test_loader)
        self.logging(epoch_loss, 'test')
        self.metrics.confusion_matrix_map(0, 'test', 1)
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