"""
Tensorboard logger code referenced from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/
Other helper functions:
https://github.com/cs230-stanford/cs230-stanford.github.io
"""

import json
import logging
import os
import shutil

import torch
from matplotlib import pyplot as plt
# import tensorflow as tf
from sklearn import metrics
from torch.optim.lr_scheduler import _LRScheduler

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.warmup_epochs = None
        self.seed = None
        self.no_neptune = None
        self.restore_student = None
        self.exp_dir = None
        self.self_training = None
        self.batch_size = None
        self.teacher = None
        self.subset_percent = None
        self.cifar_imb_ratio = None
        self.dataset = None
        self.num_epochs = None
        self.learning_rate = None
        self.model_version = None
        self.exp_name = None
        self.restore_file = None
        self.num_classes = None
        self.regularization = None
        self.label_smoothing = None
        self.aleatory = None
        self.double_training = None
        self.self_training = None
        self.pt_teacher = None
        self.focal_loss = None
        self.isTrain = None

        # Update hyperparameters from json file
        self.update_dict_fromjson(json_path)

    def save_json(self, json_path):
        """Save parameters to json file."""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update_dict_fromjson(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def update_dict(self, add_dict):
        self.__dict__.update(add_dict)
        if '10' in self.dataset:
            self.num_classes = 10
        elif '100' in self.dataset:
            self.num_classes = 100
        else:
            assert 'Dataset Choose Error!'

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `exp_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(old_dict, json_path):
    """Saves dict of floats in json file

    Args:
        old_dict: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        new_dict = {}
        for key, value in old_dict.items():
            # if np.array(value).shape.__len__() <= 1:
            if key != 'conf_mat':
                new_dict[key] = float(value)

        json.dump(new_dict, f, indent=4)


def save_json(epoch, mr, json_path):
    import json

    new_dict = {'epoch': epoch, 'mr': mr}

    with open(json_path, 'w') as f:
        json.dump(new_dict, f)


def save_checkpoint(state, is_best, params, save_epoch_checkpoint=False):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        params:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        save_epoch_checkpoint: (bool) save checkpoint of this epoch or not
    """
    filepath = os.path.join(params.exp_dir, str(params.seed) + 'last.pth.tar')

    if not os.path.exists(params.exp_dir):
        print("Checkpoint Directory does not exist! Making directory {}".format(params.exp_dir))
        os.mkdir(params.exp_dir)

    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(params.exp_dir, str(params.seed) + 'best.pth.tar'))
    if save_epoch_checkpoint:
        epoch_file = str(params.seed) + str(state['epoch'] - 1) + '.pth.tar'
        shutil.copyfile(filepath, os.path.join(params.exp_dir, epoch_file))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def scalar2onehot(batch_labels, num_classes):
    """Convert class labels from scalars to one-hot vectors."""

    batch_size = batch_labels.shape[0]
    onehot_tensor = torch.zeros((batch_size, num_classes))

    for i in range(batch_size):
        label = batch_labels[i]  #
        onehot_tensor[i][label] = 1

    return onehot_tensor


def draw_multi_histogram(class_tf_list, class_unc_list, save_dir=None):
    """ Dram a histogram with y1 and y2 """
    # from matplotlib import pyplot as plt
    import numpy as np

    histogram_nums = len(class_unc_list)
    for i in range(histogram_nums):
        correct_index = np.where(np.array(class_tf_list[i]) == 1)
        correct_unc_list = np.array(class_unc_list[i])[correct_index].tolist()
        incorrect_index = np.where(np.array(class_tf_list[i]) == 0)
        incorrect_unc_list = np.array(class_unc_list[i])[incorrect_index].tolist()

        plt.hist(correct_unc_list, bins=100, label='Correctly Classified')
        plt.hist(incorrect_unc_list, bins=100, label='Incorrectly Classified')

        plt.xlabel("Max Uncertainty")
        plt.ylabel("Number of Samples")
        plt.legend()

        plt.title("Class{} Max Uncertainty".format(i))

        if save_dir is not None:
            fig_name = save_dir + '/Class{}_max_unc_histogram.png'.format(i)
            plt.savefig(fig_name)

        # plt.show()
        plt.clf()  # clear the current figure.


def compute_auc_for_multiclass(_true, _prob, draw_roc=False, save_dir=None):
    """Compute AUC score for multiclass classification, i.e. for each sample in specific class,
    it belongs to its original class(positive) or other class(negative).

    Args:
        _true: (N, 1) target label list consists of scalar labels.
        _prob: (N, C) predicted probability vector.
        draw_roc:
        save_dir:

    Returns:(C, 1)
        AUC score of each class.
    """
    _prob = torch.tensor(_prob)

    if _prob.ndim > 1:
        num_classes = _prob.shape[1]
        _true = scalar2onehot(_true, num_classes)
    else:
        num_classes = 1
    # if not torch.is_tensor(y_pred_prob):
    #     if type(y_pred_prob) is np.array:
    #     y_pred_prob = torch.tensor(y_pred_prob)

    auc_list = []
    for _c in range(num_classes):
        _fpr, _tpr, _thresholds = metrics.roc_curve(_true[:, _c], _prob[:, _c])
        auc_i = metrics.auc(_fpr, _tpr)
        auc_list.append(auc_i)

    if draw_roc:
        draw_roc_curve(_true=y_true, _prob=y_prob, curve_title='ROC', save_dir=save_dir)


def draw_roc_curve(_true=None, _prob=None, _fpr=None, _tpr=None, _auc=None,
                   curve_title=None, save_dir=None):
    """draw Receiver operating curve by true target (2-class label) and predicted probability."""

    """DIY draw function"""
    if len(list(_prob.size())) > 1:
        num_classes = _prob.size(1)
    else:
        num_classes = 1

    colors = ['black', 'blue', 'orange', 'green', 'red',
              'purple', 'brown', 'pink', 'cyan', 'peru']
    for _c in range(num_classes):
        _fpr, _tpr, _thresholds = metrics.roc_curve(_true[:, _c], _prob[:, _c])
        _auc = metrics.auc(_fpr, _tpr)
        plt.plot(_fpr, _tpr, color=colors[_c], label='Class{}(AUC={:.2f})'.format(_c, _auc))
        # plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        # plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体

    if curve_title is not None:
        plt.title(curve_title)
    else:
        plt.title('ROC Curve')

    if save_dir is not None:
        save_name = os.path.join(save_dir, curve_title + '.png')
        plt.savefig(save_name)

    plt.show()

    """Sklearn plot roc curve"""
    # if _true is not None:
    #     _fpr, _tpr, _thresholds = metrics.roc_curve(_true, y_pred_prob)
    #     _auc = metrics.auc(_fpr, _tpr)
    #
    # display = metrics.RocCurveDisplay(fpr=_fpr, tpr=_tpr, roc_auc=_auc, estimator_name=curve_title)
    # display.plot()
    # plt.show()
    #
    # save_name = os.path.join(save_dir, curve_title + '.png')
    # if save_dir is not None:
    # plt.savefig(save_name)
    # display.figure_.savefig(save_name)

    return 0


if __name__ == '__main__':
    # a = torch.tensor([1, 2, 3, 4, 5])
    # b = scalar2onehot(a, 6)
    # print(b)
    y_true = torch.tensor([[1, 0], [0, 1]])
    y_prob = torch.tensor([[0.3, 0.4], [0.3, 0.4]])

    draw_roc_curve(y_true, y_prob, curve_title='ROC', save_dir='.')
