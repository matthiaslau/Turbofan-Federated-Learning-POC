import os
import pandas as pd
import torch
import syft as sy
from syft import PublicGridNetwork
from syft.workers.node_client import NodeClient

from ..helper import config_helper
from ..persistence.models import SensorData, db
from ..persistence.model_manager import delete_sensor_data
from .state_handler import get_state, set_state, State
from .stats_handler import track, Stats
from ..helper import data_helper


hook = sy.TorchHook(torch)

MODEL_ID = 'turbofan'
MAINTENANCE_LEAD_TIME = 10
MAINTENANCE_GRACE_PERIOD = 5

sensor_data = None
current_row = 0

current_cycle = 0
current_prediction = None
maintenance_start_cycle = 0

shared_data = []
shared_labels = []


def handle_current_sensors(app, scheduler):
    """ Read in the next set of sensor data and handle it.

    :param app: The flask app context for accessing the db
    :param scheduler: Scheduler object to stop it in the end
    :return: None
    """
    global sensor_data, current_row, current_cycle, current_prediction, maintenance_start_cycle

    # read the sensor data and cache it
    if sensor_data is None:
        sensor_data = import_data()

    if get_state() == State.STOPPED:
        # the engine_node is not running so we start it now
        set_state(State.RUNNING)
        print("-> Engine started")

    current_row += 1

    last_run = False
    lookahead = None
    try:
        # lookahead to notice when all series will end
        lookahead = sensor_data.iloc[current_row]
    except IndexError:
        # shutdown scheduler as there is no data left
        scheduler.shutdown(wait=False)
        last_run = True

    current_sensor_values = sensor_data.iloc[current_row - 1]
    current_cycle = current_sensor_values['time_in_cycles']

    # if the next series just started check what to do
    if lookahead is None or int(lookahead['time_in_cycles'] == 1):
        current_prediction = None
        skip_cycle = handle_series_ended(app)
        maintenance_start_cycle = None
        if skip_cycle:
            return

    # create DB model and set properties
    sensor_data_object = SensorData()
    for col, val in current_sensor_values.items():
        setattr(sensor_data_object, col, val)

    with app.app_context():
        # save sensor data to DB
        db.session.add(sensor_data_object)
        db.session.commit()
        # retrieve all current sensor data
        sensor_data_all = pd.read_sql(db.session.query(SensorData).statement, db.session.bind)

    # we need at least an amount of data rows of our window size for inference
    if sensor_data_all.shape[0] >= data_helper.WINDOW_SIZE:
        current_prediction = predict_rul(sensor_data_all)

        # switch to maintenance if we predicted the failure is less than 10 cycles ahead
        if current_prediction is not None and current_prediction < MAINTENANCE_LEAD_TIME:
            if get_state() != State.MAINTENANCE:
                print("+++ Switching to maintenance +++")
                set_state(State.MAINTENANCE)
                maintenance_start_cycle = current_cycle

    print('Engine No: {:.0f}\t- Cycle: {:.0f}\t- Predicted RUL: {}'.format(
        current_sensor_values['engine_no'],
        current_sensor_values['time_in_cycles'],
        "-" if current_prediction is None else current_prediction
    ))

    if last_run:
        # all series ended, let´s stop the engine_node
        set_state(State.STOPPED)
        print("-> Engine stopped")

    return None


def handle_series_ended(app):
    """ Handle a sensor data series after it ended.

    :param app: The flask app context for accessing the db
    :return: Boolean whether the current cycle should be skipped
    """
    global current_row

    skip_cycle = False

    current_state = get_state()
    if current_state == State.RUNNING:
        # the engine is running and should now be switched to failure as the last series ended without maintenance
        set_state(State.FAILURE)
        # revert this round
        current_row -= 1
        # track unsuccessful prediction
        track(Stats.FAILURES)
        print("### Engine failure ###")
        move_current_data_to_training(app)
        skip_cycle = True
    elif current_state == State.MAINTENANCE:
        # the engine is in maintenance while the last series ended so everything is fine, we can start with the new
        # series
        set_state(State.RUNNING)
        if current_cycle - maintenance_start_cycle > MAINTENANCE_LEAD_TIME + MAINTENANCE_GRACE_PERIOD:
            # maintenance prevented failure but was too early
            print("+++ Maintenance was too early +++")
            track(Stats.PREVENTED_FAILURES_TOO_EARLY)
        else:
            # maintenance successfully prevented the failure in time
            print("+++ Maintenance successfully prevented a failure +++")
            track(Stats.PREVENTED_FAILURES)
        move_current_data_to_training(app)
    elif current_state == State.FAILURE:
        # the engine_node is already failing, we could now start over with the next series
        set_state(State.RUNNING)
        print("-> Engine fixed and up again")

    return skip_cycle


def import_data():
    """ Import the turbofan training data for this worker from disk.

    :return: The training dataset for this worker
    """
    dirname = os.getcwd()
    folder_path = os.path.join(dirname, config_helper.data_dir)
    data_path = os.path.join(folder_path, "train_data_worker_{}.txt".format(config_helper.dataset_id))
    data = pd.read_csv(data_path)
    data.set_index('time_in_cycles')

    return data


def move_current_data_to_training(app):
    """ Read all current sensor data from the db, pre-process it for model training and send it to the local worker.

    :param app: The flask app context for accessing the db
    :return: None
    """
    with app.app_context():
        # read all sensor data from DB into a pandas frame
        new_train_data = pd.read_sql(db.session.query(SensorData).statement, db.session.bind)

    # data preprocessing for training
    new_train_data = data_helper.add_rul_to_train_data(new_train_data)
    data_helper.drop_unnecessary_columns(new_train_data)
    x_train_new, y_train_new = data_helper.transform_to_windowed_data(new_train_data, with_labels=True)
    y_train_new = data_helper.clip_rul(y_train_new)

    # transform to torch tensors
    tensor_x_train_new = torch.Tensor(x_train_new)
    tensor_y_train_new = torch.Tensor(y_train_new)

    # tag the data so it can be searched within the grid
    tensor_x_train_new = tensor_x_train_new.tag("#X", "#turbofan", "#dataset").describe("The input datapoints to the turbofan dataset.")
    tensor_y_train_new = tensor_y_train_new.tag("#Y", "#turbofan", "#dataset").describe("The input labels to the turbofan dataset.")

    # send the data to the grid node
    grid_node = NodeClient(hook, address="ws://{}".format(config_helper.grid_node_address))
    shared_data.append(tensor_x_train_new.send(grid_node))
    shared_labels.append(tensor_y_train_new.send(grid_node))

    # delete current sensor data from db
    with app.app_context():
        delete_sensor_data()

    return None


def prepare_data_for_prediction(data):
    """ Perform data pre-processing for predictions.

    :param data: The data to pre-process
    :return: The prepared data as tensor
    """
    data_helper.drop_unnecessary_columns(data)
    x = data_helper.transform_to_windowed_data(data, window_limit=1, with_labels=False)

    # transform to torch tensors
    tensor_x = torch.Tensor(x)

    return tensor_x


def predict_rul(data):
    """ Predict the RUL for the given data.

    :param data: The data to predict the RUL for
    :return: The predicted RUL
    """
    # prepare the last data window for inference
    sensor_data_inference = prepare_data_for_prediction(data.copy())

    # predict the current RUL
    my_grid = PublicGridNetwork(hook, "http://{}".format(config_helper.grid_gateway_address))

    prediction = None
    try:
        prediction = int(my_grid.run_remote_inference(MODEL_ID, sensor_data_inference))
    except RuntimeError:
        print('Model "{}" does not exist.'.format(MODEL_ID))
        pass

    return prediction
