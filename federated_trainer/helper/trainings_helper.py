import asyncio
import torch
import numpy as np
from torch import nn, optim

from syft.frameworks.torch.fl.utils import federated_avg

from .data_helper import get_test_data
from . import config_helper as config


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


def count_worker_data(data):
    total = 0
    for i in range(len(data)):
        total += data[i].shape[0]

    return total


async def train(model, data, labels, criterion, epochs):
    """ Train the model with the given data in a federated way.

    :param model: Model to train
    :param data: Data to use for training
    :param labels: Labels to use for loss calculation
    :param criterion: Criterion to use for loss calculations
    :param epochs: Number of epochs to run for
    :return: A tuple of the model and the loss of this training round
    """
    model.train()

    running_loss = 0
    worker_models = {}

    # asynchronously perform the training on each worker
    results = await asyncio.gather(
        *[
            # retrieve the worker from the first data sample, we expect all data is on the same worker
            train_on_worker(
                data[i][0].location, model.copy(), criterion, data[i], labels[i], epochs, verbose=True
            )
            for i in range(len(data))
        ]
    )

    for worker_id, worker_model, worker_loss in results:
        worker_models[worker_id] = worker_model
        running_loss += worker_loss
        rmse = get_model_error(worker_model)
        print('Worker: {}\tLoss: {:.4f}\tRMSE: {:7.2f}'.format(worker_id, worker_loss, rmse))

    round_loss = running_loss / len(data)
    # average all worker models
    model = federated_avg(worker_models)

    # TODO: add EarlyStopping

    return model, round_loss


async def train_on_worker(worker, model, criterion, data, labels, epochs, verbose=False):
    """ Train a round on the given worker node.

    :param worker: worker to train on
    :param model: model to use for training
    :param criterion: Criterion to use for loss calculations
    :param data: training data
    :param labels: training labels
    :param epochs: epochs to train for
    :param verbose: whether the epoch losses should be printed
    :return: a tuple of the worker id, the model and the loss
    """
    worker_loss = 0

    # initialize an dedicated optimizer for every worker to prevent errors with adams momentum
    optimizer = optim.Adam(model.parameters())
    # send the model to the worker
    model.send(worker)

    data_count = count_worker_data(data)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        for j in range(len(data)):
            optimizer.zero_grad()
            output = model(data[j])

            loss = criterion(output, labels[j])
            loss.backward()
            optimizer.step()

            loss = loss.get()
            epoch_loss += loss.item() * data[j].shape[0]

        epoch_loss = epoch_loss / data_count
        worker_loss = epoch_loss
        if verbose and epoch % 10 == 0:
            print('Epoch: {}/{}\tloss: {:.4f}'.format(epoch, epochs, epoch_loss))

    # get the updated model and the loss back from the worker
    model.get()

    return worker.id, model, worker_loss


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


def start_federated_training(model, train_data, train_labels):
    """ Start federated training.

    :param model: Model to train
    :param train_data: Training data
    :param train_labels: Training labels
    :return: a tuple of the trained model and the loss of this round
    """
    torch.manual_seed(1)
    np.random.seed(51)

    criterion = nn.L1Loss()  # mae

    model, round_loss = asyncio.run(
        train(model, train_data, train_labels, criterion, config.epochs)
    )

    return model, round_loss


def get_model_error(model):
    """ Calculate the RMSE for the given model and test data.

    :param model: Model to use
    :param test_data: test data
    :param test_labels: test labels
    :return: RMSE score
    """
    model.eval()

    test_data, test_labels = get_test_data()

    with torch.no_grad():
        output = model(test_data)
        rmse = root_mean_squared_error(test_labels, output)

    return rmse
