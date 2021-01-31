import torch
import math
import numpy as np


class Result(object):
    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.loss = 0

    def set_to_worst(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.loss = 0

    def update(self, accuracy, precision, recall, loss):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.loss = loss

    def evaluate(self, predictions, labels, loss, mask=None):
        predictions = np.argmax(predictions.cpu().numpy(), axis=-1)
        labels = labels.cpu().numpy()

        if mask is not None:
            mask = torch.squeeze(mask, dim=-1).cpu().numpy().astype(np.bool)
            predictions = predictions[mask]
            labels = labels[mask]
        # pdb.set_trace()
        tp_count = np.sum((predictions == 1) & (labels == 1))
        tn_count = np.sum((predictions == 0) & (labels == 0))
        fn_count = np.sum((predictions == 0) & (labels == 1))
        fp_count = np.sum((predictions == 1) & (labels == 0))

        # ToDo: whether we should record precision
        self.accuracy = (tp_count + tn_count) / (tp_count + tn_count + fn_count + fp_count)
        self.precision = (tp_count) / (tp_count + fp_count)
        self.recall = tp_count / (tp_count + fn_count)

        self.loss = loss.cpu().numpy()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_accuracy = 0
        self.sum_precision = 0
        self.sum_recall = 0
        self.sum_loss = 0

    def update(self, result, n=1):
        self.count += n

        self.sum_accuracy += n * result.accuracy
        self.sum_precision += n * result.precision
        self.sum_recall += n * result.recall
        self.sum_loss += n * result.loss

    def average(self):
        avg = Result()
        avg.update(
            self.sum_accuracy / self.count,
            self.sum_precision / self.count,
            self.sum_recall / self.count,
            self.sum_loss / self.count
        )

        return avg
