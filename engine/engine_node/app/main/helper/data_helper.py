import numpy as np
import pandas as pd


WINDOW_SIZE = 80
RUL_CLIP_LIMIT = 110


def add_rul_to_train_data(train_data):
    """ Calculate and add the RUL to all rows in the given training data.

    :param train_data: The training data
    :return: The training data with added RULs
    """
    # retrieve the max cycles per engine_node: RUL
    train_rul = pd.DataFrame(train_data.groupby('engine_no')['time_in_cycles'].max()).reset_index()

    # merge the RULs into the training data
    train_rul.columns = ['engine_no', 'max']
    train_data = train_data.merge(train_rul, on=['engine_no'], how='left')

    # add the current RUL for every cycle
    train_data['RUL'] = train_data['max'] - train_data['time_in_cycles']
    train_data.drop('max', axis=1, inplace=True)

    return train_data


def drop_unnecessary_columns(data):
    """ Drop columns with nan values, constant values and defined as irrelevant during the data analysis inplace.

    :param data: The data where the columns should be dropped
    """
    # drop the columns not needed
    cols_nan = data.columns[data.isna().any()].tolist()
    cols_const = [col for col in data.columns if len(data[col].unique()) <= 2 and not col == 'engine_no']

    # The operational settings 1 and 2 don´t have a trend and they look like random noise.
    # Sensors 11, 12, 13 could be removed due to high correlations but it should be tested.
    # The trend of sensors 9 and 14 depend on the specific engine_node. Some engines at the end
    # of life tend to increase while others tend to decrease. What is common about these
    # two sensors is that the magnitude at the end life gets amplified. We should try
    # removing both sensors.
    cols_irrelevant = ['operational_setting_1', 'operational_setting_2', 'sensor_measurement_11',
                       'sensor_measurement_12', 'sensor_measurement_13']

    # Drop the columns without or with constant data
    data.drop(columns=cols_const + cols_nan + cols_irrelevant, inplace=True)


def transform_to_windowed_data(dataset, window_limit=0, with_labels=True):
    """Transform the dataset into input windows with a label.

      Args:
          dataset (DataFrame): The dataset to tranform
          window_limit (int): The max windows to create for a data subset
          with_labels (bool): Whether labels are also included and should be processed

      Returns:
          (numpy.array, numpy.array): A tuple of features and labels.
      """
    features = []
    labels = []

    dataset = dataset.set_index('time_in_cycles')
    data_per_engine = dataset.groupby('engine_no')

    for engine_no, engine_data in data_per_engine:
        # skip if the engines cycles are too few
        if len(engine_data) < WINDOW_SIZE + window_limit - 1:
            continue

        if window_limit != 0:
            window_count = window_limit
        else:
            window_count = len(engine_data) - WINDOW_SIZE

        for i in range(0, window_count):
            # take the last x cycles where x is the window size
            start = -WINDOW_SIZE - i
            end = len(engine_data) - i
            inputs = engine_data.iloc[start:end]
            inputs = inputs.drop(['engine_no'], axis=1)

            if with_labels:
                # use the RUL of the last cycle as label
                outputs = engine_data.iloc[end - 1, -1]
                labels.append(outputs)

                inputs = inputs.drop(['RUL'], axis=1)

            features.append(inputs.values)

    features = np.array(features)
    if with_labels:
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=1)

    if with_labels:
        return features, labels
    else:
        return features


def clip_rul(labels):
    """ Clip the label data to a maximum RUL value.

    :param labels: The labels to clip
    :return: The clipped labels
    """
    return labels.clip(max=RUL_CLIP_LIMIT)
