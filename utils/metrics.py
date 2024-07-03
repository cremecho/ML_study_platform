import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import torch
np.seterr(divide='ignore', invalid='ignore')


class Metrics:
    def __init__(self, model_name, NUM_CLASSES, CLASS_LABELS, save_path):
        self.NUM_CLASSES = NUM_CLASSES
        self.CLASS_LABELS = CLASS_LABELS
        self.labels = np.arange(self.NUM_CLASSES)
        self.cm = np.zeros((self.NUM_CLASSES,) * 2)
        self.save_path = save_path
        self.model_name = model_name

    def add_batch(self, y_true, y_pred):
        if self.model_name == 'lr':
            self.add_batch_binary(y_true, y_pred)
        else:
            y_pred = torch.argmax(y_pred, dim=1)
            self.cm += confusion_matrix(y_true.cpu(), y_pred.cpu(), labels=self.labels)

    def add_batch_binary(self, y_true, y_pred):
        y_pred = torch.round(y_pred)
        y_true, y_pred = y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
        self.cm += confusion_matrix(y_true, y_pred, labels=self.labels)

    def reset(self):
        self.cm = np.zeros((self.NUM_CLASSES,) * 2)

    def confusion_matrix_map(self, epoch, mode, frequency):
        if not epoch % frequency == 0:
            return
        temp_cm = np.array(self.cm, dtype='int')
        xylabel = self.CLASS_LABELS
        plt.figure(figsize=(30, 30))
        plt.ticklabel_format(style='plain')
        plot = ConfusionMatrixDisplay(temp_cm, display_labels=xylabel, )
        plot.plot(cmap="Blues", values_format='')
        plt.savefig(os.path.join(self.save_path, 'confusion_matrix_%d_%s.png' % (epoch, mode)))
        plt.close()

    def Accuracy(self):
        Acc = np.diag(self.cm).sum() / self.cm.sum()
        return Acc

    def Accuracy_Class(self):
        Acc = np.diag(self.cm) / self.cm.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Sensitivity(self, class_idx):
        true_positives = self.cm[class_idx, class_idx]
        false_negatives = np.sum(self.cm[class_idx, :]) - true_positives
        sensitivity = true_positives / (true_positives + false_negatives)
        return sensitivity

    def Specificity(self, class_idx):
        true_negatives = np.sum(self.cm) - np.sum(self.cm[class_idx, :]) - np.sum(self.cm[:, class_idx]) + self.cm[class_idx, class_idx]
        false_positives = np.sum(self.cm[:, class_idx]) - self.cm[class_idx, class_idx]
        specificity = true_negatives / (true_negatives + false_positives)
        return specificity

    def Precision(self, class_idx):
        true_positives = self.cm[class_idx, class_idx]
        false_positives = np.sum(self.cm[:, class_idx]) - self.cm[class_idx, class_idx]
        precision = true_positives / (true_positives + false_positives)
        return precision

    def F1(self, class_idx):
        precision = self.Precision(class_idx)
        recall = self.Sensitivity(class_idx)
        f1 = 2 * precision * recall / (precision + recall)
        if np.isnan(f1):
            f1 = 0.
        return f1

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.cm) / (
                    np.sum(self.cm, axis=1) + np.sum(self.cm, axis=0) -
                    np.diag(self.cm))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.cm, axis=1) / np.sum(self.cm)
        iu = np.diag(self.cm) / (
                    np.sum(self.cm, axis=1) + np.sum(self.cm, axis=0) -
                    np.diag(self.cm))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU