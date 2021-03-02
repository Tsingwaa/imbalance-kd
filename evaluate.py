"""Evaluates the model"""

from sklearn import metrics


def evaluate(model, dataloader):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    """

    # set model to evaluation mode
    import numpy as np

    model.eval()
    y_true = []
    y_pred = []

    # compute metrics over the dataset test
    for data_batch, labels_batch in dataloader:
        data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        # data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        _, predicted = output_batch.max(1)

        y_true.extend(list(labels_batch.cpu().numpy()))
        y_pred.extend(list(predicted.cpu().numpy()))

    mean_recall = metrics.balanced_accuracy_score(y_true, y_pred)
    mean_recall = round(mean_recall, 4)  # save 4 effective number

    class_recall = metrics.recall_score(y_true, y_pred, average=None)  # output each class recall
    class_recall = np.round(class_recall, decimals=2).tolist()

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    return {'cr': class_recall, 'mr': mean_recall, 'conf_mat': confusion_matrix}


def evaluate_kd(model, dataloader):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    """

    # set model to evaluation mode
    import numpy as np

    model.eval()
    y_true = []
    y_pred = []

    # compute metrics over the dataset
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        # move to GPU if available
        data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()

        # compute model output
        output_batch = model(data_batch)

        _, predicted = output_batch.max(1)
        y_true.extend(list(labels_batch.cpu().numpy()))
        y_pred.extend(list(predicted.cpu().numpy()))

    mean_recall = metrics.balanced_accuracy_score(y_true, y_pred)
    mean_recall = round(mean_recall, 4)  # save 4 effective number

    class_recall = metrics.recall_score(y_true, y_pred, average=None)  # output each class recall
    class_recall = np.round(class_recall, decimals=2).tolist()

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # logging.info("========> Class Recall: {cr}\t Mean:{mr:.2%}".format(cr=class_recall, mr=mean_recall))

    return {'cr': class_recall, 'mr': mean_recall, 'conf_mat': confusion_matrix}


def evaluate_unc(model, dataloader):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
    """

    # set model to evaluation mode
    import numpy as np
    model.eval()
    y_true = []
    y_pred = []

    # compute metrics over the dataset test
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        # data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output, sigma = model(inputs)
        _, predicted = output.max(1)

        y_true.extend(list(targets.cpu().numpy()))
        y_pred.extend(list(predicted.cpu().numpy()))

    mean_recall = metrics.balanced_accuracy_score(y_true, y_pred)
    mean_recall = round(mean_recall, 4)  # save 4 effective number

    class_recall = metrics.recall_score(y_true, y_pred, average=None)  # output each class recall
    class_recall = np.round(class_recall, decimals=2).tolist()

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # logging.info("========> Eval Mean Recall: {mr:.2%}".format(mr=mean_recall))

    return {'cr': class_recall, 'mr': mean_recall, 'conf_mat': confusion_matrix}
