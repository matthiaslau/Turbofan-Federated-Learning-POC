import torch
import numpy as np
from torch import nn, optim

from .data_helper import get_data_loader


history = {}


def root_mean_squared_error(y_true, y_pred):
    """ RMSE implementation.

    :param y_true: The correct labels
    :param y_pred: The predicted labels
    :return: RMSE score
    """
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def data_result_size(data):
    """ Count all samples in all objects for all workers in a grid search response.

    :param data: Grid search response data
    :return: Total number of data samples
    """
    total = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            total += data[i][j].shape[0]

    return total


def train(model, data, labels, criterion):
    """ Train the model with the given data in a federated way.

    :param model: Model to train
    :param data: Data to use for training
    :param labels: Labels to use for loss calculation
    :param criterion: Criterion to use for loss calculations
    :return: Loss of this epoch
    """
    model.train()

    epoch_total = data_result_size(data)

    running_loss = 0

    for i in range(len(data)):
        # initialize an dedicated optimizer for every worker to prevent errors with adams momentum
        optimizer = optim.Adam(model.parameters())

        for j in range(len(data[i])):
            # check the location of the data and send the model there
            worker = data[i][j].location
            model.send(worker)

            # train one step
            optimizer.zero_grad()
            output = model(data[i][j])

            loss = criterion(output, labels[i][j])
            loss.backward()
            optimizer.step()

            # get the updated model and the loss back from the worker
            model.get()
            loss = loss.get()

            running_loss += loss.item() * data[i][j].shape[0]

    epoch_loss = running_loss / epoch_total

    # TODO: add EarlyStopping

    return epoch_loss


def test(model, test_loader, criterion):
    """ Test the model.

    :param model: The model to test
    :param test_loader: DataLoader with the test data
    :param criterion: Criterion to use for loss calculation
    :return: Loss of the test
    """
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)

            test_loss += criterion(output, target).item() * data.size(0)

    test_loss /= len(test_loader.dataset)

    return test_loss


def start_federated_training(model, train_data, train_labels, val_data, val_labels, epochs):
    """ Start federated training.

    :param model: Model to train
    :param train_data: Training data
    :param train_labels: Training labels
    :param val_data: Validation data
    :param val_labels: Validation labels
    :param epochs: Number of epochs
    :return: The trained model
    """
    global history

    torch.manual_seed(1)
    np.random.seed(51)

    criterion = nn.L1Loss()  # mae

    # manually track the losses
    history['epoch'] = []
    history['loss'] = []
    history['val_loss'] = []

    val_loader = get_data_loader(val_data, val_labels)

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_data, train_labels, criterion)
        val_loss = test(model, val_loader, criterion)

        print('Epoch: {}/{}\tloss: {:.4f}\tval_loss: {:.4f}'.format(epoch, epochs, train_loss, val_loss))

        history['epoch'].append(epoch)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    return model


def get_model_error(model, test_data, test_labels):
    """ Calculate the RMSE for the given model and test data.

    :param model: Model to use
    :param test_data: test data
    :param test_labels: test labels
    :return: RMSE score
    """
    model.eval()

    with torch.no_grad():
        output = model(test_data)
        rmse = root_mean_squared_error(test_labels, output)

    return rmse
