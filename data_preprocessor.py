#!/bin/env python
"""
    Turbofan data preprocessor to download, prepare and split the NASA turbofan data set in needed parts.
"""
import os
import pandas as pd
import random
import argparse
import requests
import zipfile
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Run Data Preprocessor.")

parser.add_argument(
    "--turbofan_dataset_id",
    type=str,
    help="ID of the turbofan dataset, e.g. FD001.",
    default=os.environ.get("TURBOFAN_DATASET_ID", "FD001"),
)

parser.add_argument(
    "--engine_percentage_initial",
    type=int,
    help="Percentage of train engines used for initial model training.",
    default=os.environ.get("ENGINE_PERCENTAGE_INITIAL", 10),
)

parser.add_argument(
    "--engine_percentage_val",
    type=int,
    help="Percentage of test engines used for cross validation.",
    default=os.environ.get("ENGINE_PERCENTAGE_VAL", 50),
)

parser.add_argument(
    "--worker_count",
    type=int,
    help="Number of workers used.",
    default=os.environ.get("WORKER_COUNT", 6),
)

parser.add_argument(
    "--no_download",
    help="Dont Download datasets when argument is present.",
    action="store_true",
)


def download_datasets():
    """ Download and unzip the NASA turbofan dataset. """
    file_name = "data.zip"
    url = "http://ti.arc.nasa.gov/c/6/"
    response = requests.get(url, stream=True)
    with open(file_name, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

    my_zip = zipfile.ZipFile('data.zip')
    storage_path = './data/'
    for file in my_zip.namelist():
        if my_zip.getinfo(file).filename.endswith('.txt'):
            my_zip.extract(file, storage_path)

    os.remove(file_name)


def import_data(dataset_id):
    """ Import the turbofan training and test data and the test RUL values from the data files.

    :param dataset_id: The dataset from turbofan to import
    :return: A tuple with the training dataset, the test dataset and the test rul data
    """
    # define the columns in the dataset
    operational_settings = ['operational_setting_{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['sensor_measurement_{}'.format(i + 1) for i in range(23)]
    cols = ['engine_no', 'time_in_cycles'] + operational_settings + sensor_columns

    # load the data
    dirname = os.getcwd()
    folder_path = os.path.join(dirname, 'data')

    train_path = os.path.join(folder_path, 'train_{}.txt'.format(dataset_id))
    train_data = pd.read_csv(train_path, delim_whitespace=True, header=None, names=cols)
    train_data.set_index('time_in_cycles')
    test_path = os.path.join(folder_path, 'test_{}.txt'.format(dataset_id))
    test_data = pd.read_csv(test_path, delim_whitespace=True, header=None, names=cols)
    test_data.set_index('time_in_cycles')
    test_data_rul_path = os.path.join(folder_path, 'RUL_{}.txt'.format(dataset_id))
    test_data_rul = pd.read_csv(test_data_rul_path, delim_whitespace=True, header=None, names=['RUL'])

    return train_data, test_data, test_data_rul


def save_data(train_data_initial, train_data_worker, test_data_val, test_data_test):
    """ Save the prepared data sets into csv files.

    :param train_data_initial: The data for initial training to save
    :param train_data_worker: An array of data for every worker to save
    :param test_data_val: The validation data to save
    :param test_data_test: The test data to save
    :return: None
    """
    dirname = os.getcwd()
    folder_path = os.path.join(dirname, 'data')

    train_data_initial_path = os.path.join(folder_path, 'train_data_initial.txt')
    train_data_initial.to_csv(train_data_initial_path, index=False)

    for index, data in enumerate(train_data_worker):
        train_data_worker_path = os.path.join(folder_path, 'train_data_worker_{}.txt'.format(index + 1))
        data.to_csv(train_data_worker_path, index=False)

    test_data_val_path = os.path.join(folder_path, 'test_data_val.txt')
    test_data_val.to_csv(test_data_val_path, index=False)

    test_data_test_path = os.path.join(folder_path, 'test_data_test.txt')
    test_data_test.to_csv(test_data_test_path, index=False)


def add_rul_to_test_data(test_data, test_data_rul):
    """ Enhance each row in the test data with the RUL. This is done inplace.

    :param test_data: The test data to enhance
    :param test_data_rul: The final RUL values for the engines in the test data
    :return: None
    """
    # prepare the RUL file data
    test_data_rul['engine_no'] = test_data_rul.index + 1
    test_data_rul.columns = ['final_rul', 'engine_no']

    # retrieve the max cycles in the test data
    test_rul_max = pd.DataFrame(test_data.groupby('engine_no')['time_in_cycles'].max()).reset_index()
    test_rul_max.columns = ['engine_no', 'max']

    test_data = test_data.merge(test_data_rul, on=['engine_no'], how='left')
    test_data = test_data.merge(test_rul_max, on=['engine_no'], how='left')

    # add the current RUL for every cycle
    test_data['RUL'] = test_data['max'] + test_data['final_rul'] - test_data['time_in_cycles']
    test_data.drop(['max', 'final_rul'], axis=1, inplace=True)

    return test_data


def split_train_data_by_engines(train_data, engine_percentage_initial, worker_count):
    """ Groups the train data by engines and split it into subsets for initial training and for each worker.

    :param train_data: The full training data set
    :param engine_percentage_initial: The percentage of engines to take for initial training
    :param worker_count: The number of workers to prepare data sets for
    :return: A tuple with the initial training data and an array of the worker data
    """
    train_data_per_engines = train_data.groupby('engine_no')
    train_data_per_engines = [train_data_per_engines.get_group(x) for x in train_data_per_engines.groups]
    random.shuffle(train_data_per_engines)

    # split into data for initial training and data for the worker nodes
    engine_count_initial = int(len(train_data_per_engines) * engine_percentage_initial / 100)
    train_data_initial = pd.concat(train_data_per_engines[:engine_count_initial])
    train_data_worker_all = train_data_per_engines[engine_count_initial:]

    train_data_worker = []
    engine_count_worker = int((len(train_data_per_engines) - engine_count_initial) / worker_count)

    # split worker data into the data sets for every single worker
    for i in range(worker_count):
        start = i * engine_count_worker
        end = start + engine_count_worker
        train_data_worker.append(pd.concat(train_data_worker_all[start:end]))

    return train_data_initial, train_data_worker


def split_test_data_by_engines(test_data, engine_percentage_val):
    """ Groups the train data by engines and split it into a subset for validation and one for testing.

    :param test_data: The full test data set
    :param engine_percentage_val: The percentage of engines to take for validation
    :return: A tuple of the validation and the test data
    """
    test_data_per_engines = test_data.groupby('engine_no')
    test_data_per_engines = [test_data_per_engines.get_group(x) for x in test_data_per_engines.groups]
    random.shuffle(test_data_per_engines)

    engine_count_val = int(len(test_data_per_engines) * engine_percentage_val / 100)
    test_data_val = pd.concat(test_data_per_engines[:engine_count_val])
    test_data_test = pd.concat(test_data_per_engines[engine_count_val:])

    return test_data_val, test_data_test


if __name__ == "__main__":
    args = parser.parse_args()

    # read in the arguments
    dataset_id = args.turbofan_dataset_id
    engine_percentage_initial = args.engine_percentage_initial
    engine_percentage_val = args.engine_percentage_val
    worker_count = args.worker_count
    no_download = args.no_download

    if not no_download:
        print('Starting download of datasets')
        download_datasets()

    print("\n##########")
    print("Importing data for data set {}".format(dataset_id))
    train_data, test_data, test_data_rul = import_data(dataset_id)
    test_data = add_rul_to_test_data(test_data, test_data_rul)

    print("Splitting training data into subsets")
    print("Using {}% data for initial training".format(engine_percentage_initial))
    print("Creating subsets for {} worker".format(worker_count))
    train_data_initial, train_data_worker = split_train_data_by_engines(
        train_data,
        engine_percentage_initial,
        worker_count
    )
    print("Splitting test data into sets for validation and testing")
    print("Using {}% data for validation".format(engine_percentage_val))
    test_data_val, test_data_test = split_test_data_by_engines(test_data, engine_percentage_val)

    print("Saving data sets")
    save_data(train_data_initial, train_data_worker, test_data_val, test_data_test)
    print("Done")
    print("##########")
