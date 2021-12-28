#-*- coding:utf-8 -*-

# Author:james Zhang
"""
    utilities file for Pytorch models
"""

from functools import wraps
import traceback
from _thread import start_new_thread
import torch.multiprocessing as mp
import torch
import datetime
import numpy as np
from dgl.dataloading.pytorch import NodeDataLoader

# class MultiGraphNodeDataLoader(NodeDataLoader):
#     def


class EarlyStopping(object):
    def __init__(self, patience, filename):
        dt = datetime.datetime.now()
        self.filename = filename
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, acc, model):
        if self.best_acc is None:
            self.best_acc = acc
            self.save_checkpoint(model)
        elif acc < self.best_acc:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if acc >= self.best_acc:
                self.save_checkpoint(model)
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))




class early_stopper(object):

    def __init__(self, patience=10, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.best_value = None
        self.is_earlystop = False
        self.count = 0
        self.val_preds = []
        self.val_logits = []

    def earlystop(self, loss, preds, logits):

        value = -loss

        if self.best_value is None:
            self.best_value = value
            self.val_preds = preds
            self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStoper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.val_preds = preds
            self.val_logits = logits
            self.count = 0


# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    用于Pytorch的OpenMP的包装方法。Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function
