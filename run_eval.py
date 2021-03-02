"""This script aimed to run basic evaluate experiments after training to analyze some property."""
import logging
import os
import random

import numpy as np
import torch
from sklearn import metrics
from torch.nn import functional as F
from tqdm import tqdm

import utils
from dataset import data_loader
from model import model_utils


def run_eval():
    # Set hyperparameter
    exp_dir = 'experiments/imbalance_experiments/base_resnet18_unc/CF10_0.01_Sq2LSCE_0'
    params = utils.Params(os.path.join(exp_dir, 'params.json'))
    params.exp_dir = exp_dir
    params.num_classes = 10
    params.batch_size = 128
    params.isTrain = False
    params.augmentation = 'no'

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    np.random.seed(230)
    torch.cuda.manual_seed(230)

    torch.cuda.empty_cache()
    torch.set_printoptions(linewidth=200)
    np.set_printoptions(linewidth=200, suppress=True)

    # Set log
    utils.set_logger(os.path.join(exp_dir, 'test.log'))

    # Create test dataloader
    logging.info("\n========> Loading Datasets: " + params.dataset + " " + str(params.cifar_imb_ratio))
    tr_loader = data_loader.fetch_dataloader('train', params)
    ts_loader = data_loader.fetch_dataloader('test', params)
    dataloader = tr_loader if params.isTrain else ts_loader

    # build model and optimizer
    model = model_utils.get_model(params)
    optimizer = model_utils.get_optimizer(model, params)
    logging.info("\n========> Loading Model({}) and Optimizer".format(params.model_version))
    # load checkpoint for model and optimizer
    restore_path = os.path.join(exp_dir, 'best.pth.tar')
    utils.load_checkpoint(restore_path, model, optimizer)

    # eval step
    model.eval()
    y_true = []
    y_pred = []
    # num_list = [0 for p in range(params.num_classes)]
    # all_mean_list = [[] for p in range(params.num_classes)]  # collect all mu by category
    # all_sigma_list = [[] for p in range(params.num_classes)]  # collect all sigma by category
    all_mean_list = []
    all_sigma_list = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(dataloader, total=len(dataloader))):
            inputs, targets = inputs.cuda(), targets.cuda()

            mean, sigma = model(inputs)
            # mean = model(inputs)
            mean = F.softmax(mean, dim=1)

            mean_list = mean.cpu().numpy().tolist()
            all_mean_list.extend(mean_list)

            sigma_list = sigma.cpu().numpy().tolist()
            all_sigma_list.extend(sigma_list)

            _, predicted = mean.max(1)

            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    """Compute confusion matrix, class recall, mean recall and auroc score."""
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    logging.info("========> Confusion Matrix: \n{}".format(confusion_matrix))

    class_recall = metrics.recall_score(y_true, y_pred, average=None)  # output each class recall
    class_recall = np.round(class_recall, decimals=2)
    logging.info("========> Class-level Recall: {}".format(class_recall.tolist()))

    mean_recall = metrics.balanced_accuracy_score(y_true, y_pred)
    mean_recall = round(mean_recall, 4)
    logging.info("========> Mean Recall: {mr:.2%}".format(mr=mean_recall))

    # get max uncertainty and softmax score by category according to whether the sample is classified correctly or not.
    class_tf_list, class_max_sfm_list, class_max_unc_list = get_max_score_categorily(y_true, y_pred, all_mean_list,
                                                                                     all_sigma_list, params.num_classes)

    # Draw histogram for each class with max uncertainty
    utils.draw_multi_histogram(class_tf_list, class_max_unc_list, save_dir=params.exp_dir)

    class_auc_max_sfm = []
    class_auc_max_unc = []
    for cls in range(params.num_classes):
        auc_max_sfm = metrics.roc_auc_score(class_tf_list[cls], class_max_sfm_list[cls])
        class_auc_max_sfm.append(auc_max_sfm)
        auc_max_unc = metrics.roc_auc_score(class_tf_list[cls], class_max_unc_list[cls])
        class_auc_max_unc.append(auc_max_unc)

    allclass_auc_max_sfm, allclass_auc_max_unc = get_all_max_unc_histogram(y_true, y_pred, all_mean_list,
                                                                           all_sigma_list, save_dir=params.exp_dir)

    class_auc_max_sfm = np.round(class_auc_max_sfm, decimals=2)
    class_auc_max_unc = np.round(class_auc_max_unc, decimals=2)
    allclass_auc_max_sfm = np.round(allclass_auc_max_sfm, decimals=4)
    allclass_auc_max_unc = np.round(allclass_auc_max_unc, decimals=4)
    logging.info('========> Class-level AUC with max softmax: {}'.format(class_auc_max_sfm.tolist()))
    logging.info('========> Class-level AUC with max uncertainty: {}\n'.format(class_auc_max_unc.tolist()))
    logging.info('========> All Class AUC with max softmax: {:.2%}'.format(allclass_auc_max_sfm))
    logging.info('========> All Class AUC with max uncertainty: {:.2%}\n'.format(allclass_auc_max_unc))

    # the class_mu_list is not equal num. so compute mean by each class
    # mu_avg = [[] for p in range(params.num_classes)]
    # sigma_avg = [[] for p in range(params.num_classes)]
    # for k in range(params.num_classes):
    #     class_mu_all = np.array(all_mean_list[k])
    #     class_mu_mean = np.mean(class_mu_all, axis=0)
    #     mu_avg[k].extend(class_mu_mean)
    #
    #     class_sigma_all = np.array(all_sigma_list[k])
    #     class_sigma_mean = np.mean(class_sigma_all, axis=0)
    #     sigma_avg[k].extend(class_sigma_mean)
    #
    # mu_avg = np.round(mu_avg, decimals=2)
    # sigma_avg = np.round(sigma_avg, decimals=2)
    # logging.info("========> Model Output:\nmu_avg:\n{}\nsigma_avg:\n{}".format(mu_avg, sigma_avg))


def get_max_score_categorily(_true, _pred, _mean, _sigma, num_classes):
    class_tf_list = [[] for p in range(num_classes)]
    class_max_sfm_list = [[] for p in range(num_classes)]  # category, true/false, mean vec.
    class_max_unc_list = [[] for p in range(num_classes)]  # category, true/false, uncertainty vec.

    for i, true_label in enumerate(_true):
        # if the sample is classified correctly.
        class_tf_list[true_label].append(int(true_label == _pred[i]))
        class_max_sfm_list[true_label].append(max(_mean[i]))
        class_max_unc_list[true_label].append(max(_sigma[i]))

    # class_tf_array = np.array(class_tf_list)
    # class_max_sfm_array = np.array(class_max_sfm_list)
    # class_max_unc_array = np.array(class_max_unc_list)

    return class_tf_list, class_max_sfm_list, class_max_unc_list


def get_all_max_unc_histogram(_true, _pred, _mean, _sigma, save_dir=None):
    """get all correctly/incorrectly max uncertainty and draw a histogram"""
    from matplotlib import pyplot as plt
    all_tf_list = []
    all_max_sfm_list = []
    all_max_unc_list = []

    for i, true_label in enumerate(_true):
        all_tf_list.append(int(true_label == _pred[i]))
        all_max_sfm_list.append(max(_mean[i]))
        all_max_unc_list.append(max(_sigma[i]))

    correct_index = np.where(np.array(all_tf_list) == 1)
    correct_unc_list = np.array(all_max_unc_list)[correct_index].tolist()
    incorrect_index = np.where(np.array(all_tf_list) == 0)
    incorrect_unc_list = np.array(all_max_unc_list)[incorrect_index].tolist()

    allclass_auc_max_sfm = metrics.roc_auc_score(all_tf_list, all_max_sfm_list)
    allclass_auc_max_unc = metrics.roc_auc_score(all_tf_list, all_max_unc_list)

    plt.hist(correct_unc_list, bins=100, label='Correctly Classified')
    plt.hist(incorrect_unc_list, bins=100, label='Incorrectly Classified')

    plt.xlabel("Max Uncertainty")
    plt.ylabel("Number of Samples")

    plt.title("All Class Correct/Incorrect-classified Max Uncertainty")

    if save_dir is not None:
        file_name = save_dir + '/All_class_max_unc_histogram.png'
        plt.savefig(file_name)

    # plt.show()
    plt.clf()

    return allclass_auc_max_sfm, allclass_auc_max_unc


if __name__ == '__main__':
    run_eval()
    # lis = [[] for i in range(3)]
    # lis[2].append(1)
