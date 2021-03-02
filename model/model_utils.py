#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File    : model_utils.py
@Author  : Tsingwaa Tsang
@Contact : zengchh3@mail2.sysu.edu.cn
@Site    : 
@License :   

@Modify Time        @Version    @Description
------------        --------    -----------
12/31/20 8:22 PM     1.0        Store fetch_utils functions
"""


def get_model(params):
    from torchvision import models
    from model import alexnet, cifar_resnet, densenet, googlenet, mobilenetv2, net, resnet, resnext, shufflenetv2

    model = None

    if params.model_version == "cnn":
        model = net.Net(params).cuda()

    elif params.model_version == "mobilenet_v2":
        model = mobilenetv2.mobilenetv2(class_num=params.num_classes).cuda()

    elif params.model_version == "shufflenet_v2":
        model = shufflenetv2.shufflenetv2(class_num=params.num_classes).cuda()

    elif params.model_version == "alexnet":
        model = alexnet.alexnet(num_classes=params.num_classes).cuda()

    elif params.model_version == "vgg19":
        model = models.vgg19_bn(num_classes=params.num_classes).cuda()

    elif params.model_version == "googlenet":
        model = googlenet.GoogleNet(num_classes=params.num_classes).cuda()

    elif params.model_version == "densenet121":
        model = densenet.densenet121(num_classes=params.num_classes).cuda()

    elif params.model_version == "resnet18":
        model = resnet.ResNet18(num_classes=params.num_classes).cuda()

    elif params.model_version == "resnet50":
        model = resnet.ResNet50(num_classes=params.num_classes).cuda()

    elif params.model_version == "resnet101":
        model = resnet.ResNet101(num_classes=params.num_classes).cuda()

    elif params.model_version == "resnet152":
        model = resnet.ResNet152(num_classes=params.num_classes).cuda()

    elif params.model_version == "resnext29":
        model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=params.num_classes).cuda()

    elif params.model_version == "resnet18_unc":
        model = resnet.ResNet18_UNC(num_classes=params.num_classes).cuda()

    elif params.model_version == "resnet32":
        model = cifar_resnet.resnet32(num_classes=params.num_classes).cuda()

    elif params.model_version == "resnet32_unc":
        model = cifar_resnet.resnet32_unc(num_classes=params.num_classes).cuda()

    return model


def get_teacher_model(params):
    from torch import nn
    from torchvision import models
    from model import alexnet, cifar_resnet, densenet, googlenet, mobilenetv2, resnet, resnext, shufflenetv2

    teacher_model = None
    teacher_checkpoint = None
    if params.teacher == "resnet18":
        teacher_model = resnet.ResNet18(num_classes=params.num_classes).cuda()
        if params.cifar_imb_ratio == 0.1:
            teacher_checkpoint = './experiments/imbalance_experiments/base_resnet18/CF10_0.1/' + \
                                 str(params.seed) + 'best.pth.tar'
        else:
            teacher_checkpoint = './experiments/imbalance_experiments/base_resnet18/CF10_0.01/' + \
                                 str(params.seed) + 'best.pth.tar'

        # if params.poor_teacher:  # poorly-trained teacher for Defective KD experiments
        #     teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet18/0.pth.tar'

        # teacher_model = teacher_model.cuda()

    elif params.teacher == "resnet32":
        teacher_model = cifar_resnet.resnet32(num_classes=params.num_classes).cuda()
        if params.cifar_imb_ratio == 0.1:
            teacher_checkpoint = './experiments/imbalance_experiments/base_resnet32/CF10_0.1/' + \
                                 str(params.seed) + 'best.pth.tar'
        else:
            teacher_checkpoint = './experiments/imbalance_experiments/base_resnet32/CF10_0.01/' + \
                                 str(params.seed) + 'best.pth.tar'
        # teacher_model = teacher_model.cuda()

    elif params.teacher == "resnet32_unc":
        teacher_model = cifar_resnet.resnet32_unc(num_classes=params.num_classes).cuda()
        if params.cifar_imb_ratio == 0.1:
            teacher_checkpoint = './experiments/imbalance_experiments/base_resnet32_unc/CF10_0.1/' + \
                                 str(params.seed) + 'best.pth.tar'
        else:
            teacher_checkpoint = './experiments/imbalance_experiments/base_resnet32_unc/CF10_0.01_GS/' + \
                                 str(params.seed) + 'best.pth.tar'
        # teacher_model = teacher_model.cuda()

    elif params.teacher == "resnet18_resample":
        teacher_model = resnet.ResNet18(num_classes=params.num_classes).cuda()
        if params.cifar_imb_ratio == 0.1:
            teacher_checkpoint = './experiments/imbalance_experiments/base_resnet18/CF100_0.1_RS/' + \
                                 str(params.seed) + 'best.pth.tar'
        else:
            teacher_checkpoint = './experiments/imbalance_experiments/base_resnet18/CF100_0.01_RS/' + \
                                 str(params.seed) + 'best.pth.tar'
        # teacher_model = teacher_model.cuda()

    elif params.teacher == "alexnet":
        teacher_model = alexnet.alexnet(num_classes=params.num_classes)
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_alexnet/' + \
                             str(params.seed) + 'best.pth.tar'
        teacher_model = teacher_model.cuda()

    elif params.teacher == "googlenet":
        teacher_model = googlenet.GoogleNet(num_classes=params.num_classes)
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_googlenet/' + \
                             str(params.seed) + 'best.pth.tar'
        teacher_model = teacher_model.cuda()

    elif params.teacher == "vgg19":
        teacher_model = models.vgg19_bn(num_classes=params.num_classes)
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_vgg19/' + \
                             str(params.seed) + 'best.pth.tar'
        teacher_model = teacher_model.cuda()

    elif params.teacher == "resnet50":
        teacher_model = resnet.ResNet50(num_classes=params.num_classes).cuda()
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet50/' + \
                             str(params.seed) + 'best.pth.tar'
        if params.pt_teacher:  # poorly-trained teacher for Defective KD experiments
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet50/' + \
                                 str(params.seed) + '.pth.tar'

    elif params.teacher == "resnet101":
        teacher_model = resnet.ResNet101(num_classes=params.num_classes).cuda()
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnet101/' + \
                             str(params.seed) + 'best.pth.tar'
        teacher_model = teacher_model.cuda()

    elif params.teacher == "densenet121":
        teacher_model = densenet.densenet121(num_classes=params.num_classes).cuda()
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_densenet121/' + \
                             str(params.seed) + 'best.pth.tar'
        # teacher = nn.DataParallel(teacher).cuda()

    elif params.teacher == "resnext29":
        teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=params.num_classes).cuda()
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnext29/' + \
                             str(params.seed) + 'best.pth.tar'
        if params.pt_teacher:  # poorly-trained teacher for Defective KD experiments
            teacher_checkpoint = 'experiments/pretrained_teacher_models/base_resnext29/' + \
                                 str(params.seed) + '.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()

    elif params.teacher == "mobilenet_v2":
        teacher_model = mobilenetv2.mobilenetv2(class_num=params.num_classes).cuda()
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_mobilenet_v2/' + \
                             str(params.seed) + 'best.pth.tar'

    elif params.teacher == "shufflenet_v2":
        teacher_model = shufflenetv2.shufflenetv2(class_num=params.num_classes).cuda()
        teacher_checkpoint = 'experiments/pretrained_teacher_models/base_shufflenet_v2/' + \
                             str(params.seed) + 'best.pth.tar'

    return teacher_model, teacher_checkpoint


def get_optimizer(model, params):
    from torch import optim

    if (params.model_version == "cnn_distill") or (params.model_version == "cnn"):
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate * (params.batch_size / 128))
    else:
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate * (params.batch_size / 128), momentum=0.9,
                              weight_decay=2e-4)

    return optimizer
