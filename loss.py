import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils


def get_loss_fn(model, params):
    if params.loss_select == 'REG':
        print('========> [ Loss KD Regularization ]')
        loss_fn = loss_kd_regularization
    elif params.loss_select == 'LS_CE':
        print('========> [ Loss Label Smoothing ]')
        loss_fn = loss_label_smoothing
    elif params.loss_select == 'Focal':
        print('========> [ Focal Loss ]')
        loss_fn = FocalLoss(2)
    elif params.loss_select == 'UNC':
        print('========> [ Uncertainty Loss ]')
        loss_fn = loss_aleatory
    else:  # choose CE loss_fn
        print('========> [ CrossEntropy Loss ]')
        loss_fn = nn.CrossEntropyLoss()
        if params.double_training:  # double training, compare to self-KD
            print(">>>>>>>>>>>>>>>>>>>>>>>>Double Training>>>>>>>>>>>>>>>>>>>>>>>>")
            checkpoint = 'experiments/pretrained_teacher_models/base_' + str(params.model_version) + '/best230.pth.tar'
            utils.load_checkpoint(checkpoint, model)
    return loss_fn


def loss_kd(y_student, y_true, y_teacher, params):
    """loss_fn function for Knowledge Distillation (KD)"""

    T = params.temperature

    # take max confidence of student model as coefficient of CE loss_fn
    # max_prob = torch.max(F.softmax(y_student, dim=1).detach(), dim=1).values
    # take max confidence of teacher model as coefficient of CE loss_fn

    loss_CE = F.cross_entropy(y_student, y_true, reduction='none')  # N*1

    y_student_log_T_softmax = F.log_softmax(y_student / T, dim=1)  # N*C
    y_teacher_T_softmax = F.softmax(y_teacher / T, dim=1)  # N*C

    # max_prob = torch.max(F.softmax(y_teacher / T, dim=1).detach(), dim=1).values
    # max_prob = torch.max(y_teacher_T_softmax.detach(), dim=1).values
    # print('Max confidence of teacher model:', torch.mean(max_prob))
    # params.maxT = torch.mean(max_prob)

    kl_div_each = F.kl_div(y_student_log_T_softmax, y_teacher_T_softmax, reduction='none') * (T * T)  # N*C

    loss_KLD = torch.mean(kl_div_each, dim=1)  # N*1

    # print('\nLoss_kd:\nCE:{}\nKLD:{}\n'.format(loss_CE.detach()[0], loss_KLD.detach()[0]))
    _loss = torch.mean((1. - params.alpha) * loss_CE + params.alpha * loss_KLD)
    # loss = torch.mean(max_prob * loss_CE + (torch.ones_like(max_prob) - max_prob) * loss_KLD)

    return _loss


def loss_kd_self(y_student, y_true, y_teacher, params):
    """ loss_fn function for self training: Tf-KD_{self} """

    T = params.temperature
    # max_prob = torch.max(F.softmax(y_student, dim=1).detach(), dim=1).values

    loss_CE = F.cross_entropy(y_student, y_true, reduction='none')  # N*1

    y_stud_log_sfm = F.log_softmax(y_student / T, dim=1)  # N*C
    y_teacher_sfm = F.softmax(y_teacher / T, dim=1)  # N*C
    kl_div_each = F.kl_div(y_stud_log_sfm, y_teacher_sfm, reduction='none') * (T * T) * params.multiplier  # N*C
    # multiplier is 1.0 in most of cases, some cases are 10 or 50
    # param.multiplier is meant to scale the magnitude.
    loss_KLD = torch.mean(kl_div_each, dim=1)  # N*1

    # print('\nloss_kd_self:\nCE:{}\nKLD:{}\n'.format(loss_CE.detach()[0], loss_KLD.detach()[0]))
    _loss = torch.mean((1. - params.alpha) * loss_CE + params.alpha * loss_KLD)
    # loss = torch.mean(max_prob * loss_CE + (1 - max_prob) * loss_KLD)

    return _loss


def loss_kd_class_weight(y_student, y_true, y_teacher, params):
    """loss_fn function for Knowledge Distillation (KD)"""
    T = params.temperature
    class_num = torch.tensor([5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50])
    reversed_class_num = 1.0 / class_num
    class_weight = reversed_class_num / reversed_class_num.sum()

    # take max confidence of student model as coefficient of CE loss_fn
    # max_prob = torch.max(F.softmax(y_student, dim=1).detach(), dim=1).values
    # take max confidence of teacher model as coefficient of CE loss_fn

    y_student_log_sm = F.log_softmax(y_student, dim=1)  # N*C
    onehot_target = utils.scalar2onehot(y_true, params.num_classes).cuda()  # N*C

    weighted_y_student_log_sm = y_student_log_sm * class_weight.cuda()  # N*C
    loss_CE = - torch.sum(onehot_target * weighted_y_student_log_sm, dim=1)  # N*1

    y_student_log_T_softmax = F.log_softmax(y_student / T, dim=1)  # N*C
    y_teacher_T_softmax = F.softmax(y_teacher / T, dim=1)  # N*C

    # max_prob = torch.max(F.softmax(y_teacher / T, dim=1).detach(), dim=1).values
    # max_prob = torch.max(y_teacher_T_softmax.detach(), dim=1).values
    # print('Max confidence of teacher model:', torch.mean(max_prob))
    # params.maxT = torch.mean(max_prob)

    kl_div_each = F.kl_div(y_student_log_T_softmax, y_teacher_T_softmax, reduction='none') * (T * T)  # N*C

    loss_KLD = torch.mean(kl_div_each, dim=1)  # N*1

    # print('\nLoss_kd:\nCE:{}\nKLD:{}\n'.format(loss_CE.detach()[0], loss_KLD.detach()[0]))
    _loss = torch.mean((1. - params.alpha) * loss_CE + params.alpha * loss_KLD)
    # loss = torch.mean(max_prob * loss_CE + (torch.ones_like(max_prob) - max_prob) * loss_KLD)

    return _loss


def loss_unc_kd(y_student, y_true, y_teacher, y_sigma, params):
    """ loss_fn function for self training: Tf-KD_{self} """

    T = params.temperature
    # max_prob = torch.max(F.softmax(y_student, dim=1).detach(), dim=1).values
    N = y_student.size(0)
    C = y_student.size(1)

    # weight = y_sigma.detach()
    weight = F.softmax(y_sigma.detach(), dim=1)

    """Weighted crossentropy """

    y_student_log_sm = F.log_softmax(y_student, dim=1)  # N*C
    onehot_target = utils.scalar2onehot(y_true, params.num_classes).cuda()  # N*C

    if params.weighted_loss == 'CE':
        weighted_y_student_log_sm = y_student_log_sm * weight  # N*C
        loss_CE = - torch.sum(onehot_target * weighted_y_student_log_sm, dim=1)  # N*1
    else:
        loss_CE = - torch.sum(onehot_target * y_student_log_sm, dim=1)  # N*1

    """Weighted KLD"""
    y_student_log_smT = F.log_softmax(y_student / T, dim=1)  # N*C
    y_teacher_smT = F.softmax(y_teacher / T, dim=1)  # N*C
    kl_div_each = F.kl_div(y_student_log_smT, y_teacher_smT, reduction='none') * (T * T)  # N*C
    if params.weighted_loss == 'KLD':
        kl_div_each = kl_div_each * weight

    loss_KLD = torch.mean(kl_div_each, dim=1)  # N*1

    _loss = torch.mean((1. - params.alpha) * loss_CE + params.alpha * loss_KLD)
    # loss = torch.mean(max_prob * loss_CE + (1 - max_prob) * loss_KLD)

    return _loss


def loss_kd_regularization(outputs, labels, params):
    """
    loss_fn function for manually-designed regularization: Tf-KD_{reg}
    """
    alpha = params.reg_alpha
    T = params.reg_temperature
    correct_prob = 0.99  # the probability for correct class in u(k)
    loss_CE = F.cross_entropy(outputs, labels)
    K = outputs.size(1)

    teacher_soft = torch.ones_like(outputs).cuda()
    teacher_soft = teacher_soft * (1 - correct_prob) / (K - 1)  # p^d(k)
    for i in range(outputs.shape[0]):
        teacher_soft[i, labels[i]] = correct_prob
    loss_soft_regu = nn.KLDivLoss()(F.log_softmax(outputs, dim=1),
                                    F.softmax(teacher_soft / T, dim=1)) * params.multiplier
    print('\nloss_kd_regularization:\nCE:{}\nKLD:{}\n'.format(loss_CE.detach(), loss_soft_regu.detach()))

    _loss = (1. - alpha) * loss_CE + alpha * loss_soft_regu

    return _loss


def weighted_crossentropy(_input, _label):
    N = _input.size(0)
    C = _input.size(1)

    class_num = torch.tensor([5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50])
    reversed_class_num = 1.0 / class_num
    class_weight = reversed_class_num / reversed_class_num.sum()

    y_student_log_sm = F.log_softmax(_input, dim=1)  # N*C
    onehot_target = utils.scalar2onehot(_label, num_classes=C).cuda()  # N*C

    weighted_y_student_log_sm = y_student_log_sm * class_weight.cuda()  # N*C
    loss_CE = - torch.sum(onehot_target * weighted_y_student_log_sm) / N  # N*1

    return loss_CE


def loss_label_smoothing(outputs, labels, alpha=0.1):
    """
    loss_fn function for label smoothing regularization
    """

    N = outputs.size(0)  # batch_size
    C = outputs.size(1)  # number of classes
    smoothed_labels = torch.full(size=(N, C), fill_value=alpha / (C - 1)).cuda()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1 - alpha)

    log_prob = torch.nn.functional.log_softmax(outputs, dim=1)
    _loss = -torch.sum(log_prob * smoothed_labels) / N

    # print('\nloss_label_smoothing:{}\n'.format(loss.detach()))

    return _loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, _input, _target):
        if _input.dim() > 2:
            _input = _input.view(_input.size(0), _input.size(1), -1)  # N,C,H,W => N,C,H*W
            _input = _input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            _input = _input.contiguous().view(-1, _input.size(2))  # N,H*W,C => N*H*W,C
        _target = _target.view(-1, 1)

        logpt = F.log_softmax(_input)
        logpt = logpt.gather(1, _target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != _input.data.type():
                self.alpha = self.alpha.type_as(_input.data)
            at = self.alpha.gather(0, _target.data.view(-1))
            logpt = logpt * Variable(at)

        _loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return _loss.mean()
        else:
            return _loss.sum()


def loss_aleatory(mean, logvar, targets):
    """Give the pred a prior knowledge of Gaussian distribution, i.e. Gaussian Likelihood function.

    Args:
        mean(batch_size, nums_classes): batch of model output logits
        targets(batch_size, nums_classes): batch of one-hot target label
        logvar(batch_size, nums_classes): batch of natural logarithm of variance
    Returns:
        Aleatory uncertainty Loss

    """

    batch_size = mean.size(0)
    num_classes = mean.size(1)

    """The original squared error"""
    # turn the scalar to one-hot tensor.
    # onehot_target = utils.scalar2onehot(targets, num_classes=num_classes).cuda()
    # squared_error = torch.pow(onehot_target - mean, 2)

    """Original squared error with label smoothing target"""
    # alpha = 0.1  # smooth level
    # smoothed_targets = torch.full(size=(batch_size, num_classes), fill_value=alpha / (num_classes - 1)).cuda()
    # smoothed_targets.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - alpha)
    # squared_error = torch.pow(smoothed_targets - mean, 2)

    """Softmax squared error with label smoothing target"""
    # alpha = 0.1  # smooth level
    # smoothed_targets = torch.full(size=(batch_size, num_classes), fill_value=alpha / (num_classes - 1)).cuda()
    # smoothed_targets.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - alpha)
    # soft_mean = F.softmax(mean, dim=1)
    # squared_error = torch.pow(smoothed_targets - soft_mean, 2)

    """Modify squared error to crossentropy loss"""
    squared_error = F.cross_entropy(mean, targets)

    """Modify squared error to label smoothing crossentropy Loss"""
    # squared_error = loss_label_smoothing(mean, targets)

    inverse_var = torch.exp(-logvar)

    batch_single_loss = 0.5 * torch.sum(inverse_var * squared_error + logvar, 1)

    batch_mean_loss = torch.mean(batch_single_loss)

    return batch_mean_loss


def KLDiv4VIB(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / mean.size(0)
    # single_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), 1) / mean.size(0)


if __name__ == '__main__':
    a = torch.rand(10, 3)
    b = torch.rand(10, 3)
    c = torch.rand(10, 3)
    loss = loss_aleatory(a, b, c)
    print(loss)
