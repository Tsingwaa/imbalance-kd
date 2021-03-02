""" Teacher free KD, main.py """

import argparse
import logging
import os
import random
import warnings

import neptune
import numpy as np
import torch

import utils
from dataset import data_loader
from loss import get_loss_fn, loss_unc_kd
from model import model_utils
from train_unc import train_and_evaluate_kd_unc, train_and_evaluate_unc

NEPTUNE_WAA_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lc" \
                    "HR1bmUuYWkiLCJhcGlfa2V5IjoiNWFjMWMwMDUtNmFlZi00ZjhkLWIwYjgtMmI4NDZkYTA3YWYzIn0="

parser = argparse.ArgumentParser()
# Experiment arguments
# parser.add_argument('--exp_dir', default='CF10_0.01_GS_2_cos', help="Directory containing params.json")
parser.add_argument('-exp', '--exp_name', default=None, required=True, help='Experiment name')
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --exp_dir \
                    containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('-kd', '--kd_experiment', action='store_true', default=False, help="flag for kd experiments")
parser.add_argument('-seed', default=230, type=int, help="Number of classes")
# data
parser.add_argument('--num_classes', default=10, type=int, help="Number of classes")
# optimizer
parser.add_argument('-T', '--temperature', default=6, type=int, help="Temperature of Knowledge Distillation")
parser.add_argument('-warm', '--warmup_epochs', type=int, default=5, help='Warm up training phase')
parser.add_argument('-wl', '--weighted_loss', default='KLD', choices=['CE', 'KLD'], help='Select which loss to weight')
parser.add_argument('--loss_select', default='UNC', choices=['CE', 'LS', 'REG', 'Focal', 'UNC'],
                    help="Choose which loss_fn to use")
# train style
parser.add_argument('--double_training', action='store_true', default=False, help="flag for double training")
parser.add_argument('--self_training', action='store_true', default=False, help="flag for self training")
parser.add_argument('--poor_teacher', action='store_true', default=False, help="flag for Defective KD")
parser.add_argument('-noNPT', '--no_neptune', action='store_true', default=False, help="flag for using neptune")


def main():
    args = parser.parse_args()
    # Load the parameters from json file
    if args.kd_experiment:
        args.exp_dir = 'experiments/kd_experiments/resnet32_unc_distill/' + args.exp_name
    else:
        args.exp_dir = 'experiments/imbalance_experiments/base_resnet32_unc/' + args.exp_name
    json_path = os.path.join(args.exp_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Create experiment recorder by neptune
    var_args = vars(args)  # transfer argparse to dict
    params.update_dict(var_args)  # update args dictionary to params object vars.

    if not params.no_neptune:
        print("Creating neptune experiment...")
        if args.kd_experiment:
            neptune.init(project_qualified_name='waa/imbalance-kd', api_token=NEPTUNE_WAA_TOKEN, )
        else:
            neptune.init(project_qualified_name='waa/imbalance', api_token=NEPTUNE_WAA_TOKEN, )
        neptune.create_experiment(name=params.exp_name, params=params.dict)

    warnings.filterwarnings("ignore")
    torch.cuda.empty_cache()
    torch.set_printoptions(linewidth=100)
    np.set_printoptions(linewidth=100)

    # Set the logger
    utils.set_logger(os.path.join(params.exp_dir, str(params.seed) + 'train.log'))

    logging.info("\n========> Loading Datasets: " + params.dataset + " " + str(params.cifar_imb_ratio))

    # fetch dataloader, considering full-set vs. sub-set scenarios
    if params.subset_percent < 1.0:  # imbalance ratio
        tr_loader = data_loader.fetch_subset_dataloader('train', params)
    else:
        tr_loader = data_loader.fetch_dataloader('train', params)

    ts_loader = data_loader.fetch_dataloader('test', params)

    # Set the random seed for reproducible experiments
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    """ Load student and teacher model according to the training style"""
    if params.teacher != 'none':

        # Specify the Student model and optimizer
        logging.info("========> Loading Student Model: {}".format(params.model_version))
        student = model_utils.get_model(params)
        optimizer = model_utils.get_optimizer(student, params)

        # load weights from restore_file if specified
        if params.restore_file is not None:
            restore_path = os.path.join(params.exp_dir, params.restore_file + '.pth.tar')
            logging.info("========> Restoring parameters from {}".format(restore_path))
            utils.load_checkpoint(restore_path, student, optimizer)

        logging.info("========> Loading Teacher Model: {}".format(params.teacher))
        teacher, teacher_checkpoint = model_utils.get_teacher_model(params)
        utils.load_checkpoint(teacher_checkpoint, teacher)

        # warmup the learning rate in the first epoch
        iter_per_epoch = len(tr_loader)
        warmup_scheduler = utils.WarmUpLR(optimizer, iter_per_epoch * params.warmup_epochs)

        # Specify Loss Function for Knowledge Distillation
        if params.self_training:  # self distillation
            logging.info('========> [ Loss_KD_self ]')
            kd_loss_fn = loss_unc_kd_self
        else:
            logging.info('========> [ Loss_KD ]')
            kd_loss_fn = loss_unc_kd

        """Train the model with Knowledge Distillation"""
        logging.info("\n>>>>>>>>>>>>>>>> [ Starting Distillation Training {} epochs... ]".format(params.num_epochs))

        train_and_evaluate_kd_unc(student, teacher, tr_loader, ts_loader, optimizer, kd_loss_fn,
                                  warmup_scheduler, params)

    else:
        """ Non-KD mode: regular training to obtain a baseline model"""
        logging.info("\n========> Loading Base Model: {}".format(params.model_version))
        model = model_utils.get_model(params)
        optimizer = model_utils.get_optimizer(model, params)

        # load weights from restore_file if specified
        if params.restore_file is not None:
            restore_path = os.path.join(params.exp_dir, params.restore_file + '.pth.tar')
            logging.info("========> Restoring parameters from {}...".format(restore_path))
            utils.load_checkpoint(restore_path, model, optimizer)

        # logging.info("========> Loading Loss function...")
        loss_fn = get_loss_fn(model, params)

        iter_per_epoch = len(tr_loader)
        warmup_scheduler = utils.WarmUpLR(optimizer, iter_per_epoch * params.warmup_epochs)

        """Train base model normally"""
        logging.info("\n>>>>>>>>>>>>>>>> [ Starting Normal Training {} epochs... ]".format(params.num_epochs))

        train_and_evaluate_unc(model, tr_loader, ts_loader, optimizer, loss_fn, warmup_scheduler, params)


if __name__ == '__main__':
    main()
