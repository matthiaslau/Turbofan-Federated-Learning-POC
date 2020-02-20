import os

import numpy as np
import pandas as pd
import torch


WINDOW_SIZE = 80
BATCH_SIZE = 210
RUL_CLIP_LIMIT = 110


def _drop_unnecessary_columns(data):
    """ Drop all columns with nan values, with constant values or defined as irrelevant during the data analysis.

    :param data: Data to drop the columns from
    :return: The given data without the columns
    """
    cols_nan = ['sensor_measurement_22', 'sensor_measurement_23']
    cols_const = [
        'operational_setting_3',
        'sensor_measurement_1',
        'sensor_measurement_5',
        'sensor_measurement_6',
        'sensor_measurement_10',
        'sensor_measurement_16',
        'sensor_measurement_18',
        'sensor_measurement_19',
        'sensor_measurement_22',
        'sensor_measurement_23'
    ]
    cols_irrelevant = [
        'operational_setting_1',
        'operational_setting_2',
        'sensor_measurement_11',
        'sensor_measurement_12',
        'sensor_measurement_13'
    ]

    return data.drop(columns=cols_const + cols_nan + cols_irrelevant)


def _transform_to_windowed_data(dataset, window_size, window_limit=0, verbose=True):
    """ Transform the dataset into input windows with a label.

    Args:
      dataset (DataFrame): The dataset to tranform.
      window_size (int): The length of the windows to create.
      window_limit (int): The max windows to create for a data subset.
      verbose (bool): Whether info on the windows should be printed.

    Returns:
      (numpy.array, numpy.array): A tuple of features and labels.
    """
    features = []
    labels = []

    dataset = dataset.set_index('time_in_cycles')
    data_per_engine = dataset.groupby('engine_no')

    for engine_no, engine_data in data_per_engine:
        # skip if the engines cycles are too few
        if len(engine_data) < window_size + window_limit - 1:
            continue

        if window_limit != 0:
            window_count = window_limit
        else:
            window_count = len(engine_data) - window_size

        for i in range(0, window_count):
            # take the last x cycles where x is the window size
            start = -window_size - i
            end = len(engine_data) - i
            inputs = engine_data.iloc[start:end]
            # use the RUL of the last cycle as label
            outputs = engine_data.iloc[end - 1, -1]

            inputs = inputs.drop(['engine_no', 'RUL'], axis=1)

            features.append(inputs.values)
            labels.append(outputs)

    features = np.array(features)
    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=1)

    if verbose:
        print("{} features with shape {}".format(len(features), features[0].shape))
        print("{} labels with shape {}".format(len(labels), labels.shape))

    return features, labels


def _clip_rul(labels):
    """ Clip the labels RUL values.

    :param labels: Labels to clip the RULs
    :return: Clipped labels
    """
    return labels.clip(max=RUL_CLIP_LIMIT)


def _load_data(filename, data_dir):
    """ Load a data file with turbofan data from disk.

    :param filename: Filename of the turbofan data files
    :param data_dir: Data directory to load the file from
    :return: Panda frame with the imported data
    """
    dirname = os.getcwd()
    folder_path = os.path.join(dirname, data_dir)

    data_path = os.path.join(folder_path, filename)
    data = pd.read_csv(data_path)
    data.set_index('time_in_cycles')

    return data


def _load_validation_data(data_dir):
    """ Load the validation data from disk.

    :param data_dir: Data directory to load the validation data from
    :return: Panda frame with the validation data
    """
    return _load_data('test_data_val.txt', data_dir)


def _load_test_data(data_dir):
    """ Load the test data from disk.

    :param data_dir: Data directory to load the test data from
    :return: Panda frame with the test data
    """
    return _load_data('test_data_test.txt', data_dir)


def _prepare_data(data):
    """ Prepare the given data by dropping unnecessary columns, creating data windows and clipping the RUL values.

    :param data: Data to prepare
    :return: A tuple of the prepares x and y tensors
    """
    data = _drop_unnecessary_columns(data)
    x, y = _transform_to_windowed_data(data, WINDOW_SIZE, verbose=False)
    y = _clip_rul(y)
    # transform to torch tensor
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)

    return tensor_x, tensor_y


def get_validation_data(data_dir):
    """ Retrieve the prepared validation data.

    :param data_dir: Data directory to load the validation data from
    :return: The prepared validation data
    """
    val_data = _load_validation_data(data_dir)
    return _prepare_data(val_data)


def get_test_data(data_dir):
    """ Retrieve the prepared test data.

    :param data_dir: Data directory to load the test data from
    :return: The prepared test data
    """
    test_data = _load_test_data(data_dir)
    return _prepare_data(test_data)


def get_data_loader(data, labels):
    """ Create a DataLoader for the given data.

    :param data: The input data
    :param labels: The labels
    :return: A DataLoader for the given data
    """
    # create dataset
    dataset = torch.utils.data.TensorDataset(data, labels)

    # create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

    return loader
