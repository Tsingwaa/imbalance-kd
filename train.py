import logging
import os

import neptune
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import utils
from evaluate import evaluate, evaluate_kd


# KD train and evaluate
def train_and_evaluate_kd(student, teacher, tr_loader, val_loader, optimizer, kd_loss_fn, warmup_scheduler, params):
    """ KD Train the model and evaluate every epoch. """

    """Set Teacher Model evaluate"""
    best_val_mr = 0.0

    teacher.eval()
    teacher_metrics = evaluate_kd(teacher, val_loader)
    logging.info("\n========> Teacher's Mean Recall: {mr:.2%}\tClass-level:{cr}\n".format(mr=teacher_metrics['mr'],
                                                                                          cr=teacher_metrics['cr']))

    lr_scheduler = MultiStepLR(optimizer, milestones=[160, 180], gamma=0.01)

    for epoch in range(1, params.num_epochs + 1):
        # if epoch > params.warmup_epochs:  # 0 is the warm up epoch
        #     lr_scheduler.step()

        adjust_learning_rate(optimizer, epoch, params)

        # log learning rate.
        lr = optimizer.param_groups[0]['lr']
        if not params.no_neptune:
            neptune.log_metric('teacher_mr', epoch, teacher_metrics['mr'])
            neptune.log_metric('lr', epoch, lr)

        # KD Train
        train_mr, train_loss = train_kd(student, teacher, optimizer, kd_loss_fn, tr_loader, warmup_scheduler, params,
                                        epoch, lr)
        # Evaluate validation set
        val_metrics = evaluate_kd(student, val_loader)
        val_mr = val_metrics['mr']
        is_best = val_mr >= best_val_mr

        # Save checkpoint
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': student.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best, params)

        # If best_eval, best_save_path
        if is_best:
            logging.info("========> Found new best result!!!")
            best_val_mr = val_mr

            # Save best val metrics in a json file in the model directory
            file_name = str(params.seed) + 'eval_best_result.json'
            best_json_path = os.path.join(params.exp_dir, file_name)
            utils.save_json(epoch, best_val_mr, best_json_path)

        logging.info("========> Val Mean Recall: {:.2%}\tClass-level: {}\n".format(val_mr, val_metrics['cr']))

        # Save latest val metrics in a json file in the model directory
        # last_json_path = os.path.join(params.exp_dir, "eval_last_result.json")
        # utils.save_dict_to_json(val_metrics, last_json_path)

        if not params.no_neptune:
            # log metrics by neptune
            neptune.log_metric('train_mr', epoch, train_mr)
            neptune.log_metric('train_loss', epoch, train_loss)
            neptune.log_metric('test_mr', epoch, val_mr)
        # log metrics by tensorboard
        # writer.add_scalar('train_mr', train_mr, epoch)
        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('test_mr', val_metrics['mr'], epoch)

        # export scalar data to JSON for external processing
    # writer.close()


# Defining train_kd functions
def train_kd(student, teacher, optimizer, kd_loss_fn, dataloader, warmup_scheduler, params, epoch, lr):
    """ KD Train the model on `num_steps` batches """

    # set model mode
    student.train()
    teacher.eval()

    loss_avg = utils.RunningAverage()
    losses = utils.AverageMeter()
    y_true = []
    y_pred = []

    # Use tqdm for progress bar
    with tqdm(desc="Epoch {}/{}".format(epoch, params.num_epochs), total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # if epoch <= params.warmup_epochs:
            #     warmup_scheduler.step()

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output, fetch teacher output, and compute KD loss_fn
            student_output_batch = student(train_batch)

            # get one batch output from teacher model
            teacher_output_batch = teacher(train_batch).cuda()
            teacher_output_batch = Variable(teacher_output_batch, requires_grad=False)

            loss = kd_loss_fn(student_output_batch, labels_batch, teacher_output_batch, params)

            # log the max temperature softmax confidence of teacher model.
            # neptune.log_metric("maxT", epoch * len(dataloader) + i, params.maxT)

            # clear previous gradients, compute gradients of all variables wrt loss_fn
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            _, predicted = student_output_batch.max(1)
            y_true.extend(list(labels_batch.cpu().numpy()))
            y_pred.extend(list(predicted.cpu().numpy()))

            # update the average loss_fn
            loss_avg.update(loss.data)
            losses.update(loss.item(), train_batch.size(0))

            t.set_postfix(loss='{:.4f}'.format(loss_avg()), lr="{:.1e}".format(lr))
            t.update()

    train_mr = metrics.balanced_accuracy_score(y_true, y_pred)

    train_mr = round(train_mr, 4)  # save 4 effective number

    return train_mr, losses.avg


# normal training
def train_and_evaluate(model, tr_loader, val_loader, optimizer, loss_fn, warmup_scheduler, params):
    """ Train the model and evaluate every epoch. """

    # if params.regularization:
    #     params.exp_dir = params.exp_dir + '/Tf-KD_regularization/'
    # elif params.label_smoothing:
    #     params.exp_dir = params.exp_dir + '/label_smoothing/'

    # dir setting, tensorboard events will save in the directory
    # log_dir = params.exp_dir + '/tensorboard/'
    # writer = SummaryWriter(log_dir=log_dir)

    best_val_mr = 0.0

    # learning rate schedulers
    # lr_scheduler = MultiStepLR(optimizer, milestones=[160, 180], gamma=0.01)
    # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=20)

    for epoch in range(1, params.num_epochs + 1):
        # if epoch > params.warmup_epochs:  # 0 is the warm up epoch
        #     lr_scheduler.step(epoch)

        adjust_learning_rate(optimizer, epoch, params)

        lr = optimizer.param_groups[0]['lr']
        if not params.no_neptune:
            neptune.log_metric('lr', epoch, lr)

        # compute number of batches in one epoch (one full pass over the training set)
        train_mr, train_loss = train(model, tr_loader, optimizer, loss_fn, warmup_scheduler, params, epoch, lr)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, val_loader)
        val_mr = val_metrics['mr']
        is_best = val_mr >= best_val_mr

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best, params)
        # If best_eval, best_save_path
        if is_best:
            logging.info("========> Found new best result!!!")
            best_val_mr = val_mr
            # Save best val metrics in a json file in the model directory
            file_name = str(params.seed) + 'eval_best_result.json'
            best_json_path = os.path.join(params.exp_dir, file_name)
            utils.save_json(epoch, best_val_mr, best_json_path)

        logging.info("========> Val Mean Recall: {:.2%}\tClass-level: {}\n".format(val_mr, val_metrics['cr']))

        # Save latest val metrics in a json file in the model directory
        # last_json_path = os.path.join(params.exp_dir, "eval_last_results.json")
        # utils.save_dict_to_json(val_metrics, last_json_path)

        if not params.no_neptune:
            # log metrics by neptune
            neptune.log_metric('train_mr', epoch, train_mr)
            neptune.log_metric('train_loss', epoch, train_loss)
            neptune.log_metric('test_mr', epoch, val_mr)
        # log metrics by tensorboard
        # writer.add_scalar('train_mr', train_mr, epoch)
        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('test_mr', val_metrics['mr'], epoch)
    #
    # writer.close()


# normal training function
def train(model, dataloader, optimizer, loss_fn, warmup_scheduler, params, epoch, lr):
    """ Normal training, without Knowledge Distillation """

    # Training mode
    model.train()

    running_avg_loss = utils.RunningAverage()
    losses = utils.AverageMeter()
    y_true = []
    y_pred = []

    # Use tqdm for progress bar
    with tqdm(desc="Epoch {}/{}".format(epoch, params.num_epochs), total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            # if epoch <= params.warmup_epochs:
            #     warmup_scheduler.step()
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            optimizer.zero_grad()
            output_batch = model(train_batch)
            if params.regularization:
                loss = loss_fn(output_batch, labels_batch, params)
            else:
                loss = loss_fn(output_batch, labels_batch)
            loss.backward()
            optimizer.step()

            _, predicted = output_batch.max(1)
            y_true.extend(list(labels_batch.cpu().numpy()))
            y_pred.extend(list(predicted.cpu().numpy()))

            # update the average loss
            running_avg_loss.update(loss.data)
            losses.update(loss.data, train_batch.size(0))

            t.set_postfix(loss='{:.4f}'.format(running_avg_loss()), lr="{:.1e}".format(lr))
            t.update()

    mean_recall = metrics.balanced_accuracy_score(y_true, y_pred)

    mean_recall = round(mean_recall, 4)  # save 4 effective number
    # print(mean_recall)

    return mean_recall, losses.avg


def adjust_learning_rate(optimizer, epoch, params):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""

    if epoch <= params.warmup_epochs:
        lr = params.learning_rate * epoch / params.warmup_epochs
    elif epoch > 270:
        lr = params.learning_rate * 1e-5  # lr=1e-6
    elif epoch > 240:
        lr = params.learning_rate * 1e-4  # lr=1e-5
    elif epoch > 200:
        lr = params.learning_rate * 1e-3  # lr=1e-4
    elif epoch > 150:
        lr = params.learning_rate * 1e-2  # lr=1e-3
    elif epoch > 100:
        lr = params.learning_rate * 1e-1  # lr=1e-2
    else:
        lr = params.learning_rate  # Init: lr=1e-1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
