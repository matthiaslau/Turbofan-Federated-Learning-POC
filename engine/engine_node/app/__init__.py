import logging

from flask import Flask

from apscheduler.schedulers.background import BackgroundScheduler
import atexit

from .main import html, db
from .main.persistence.utils import set_database_config
from .main.handler import sensor_handler
from .main.handler.sensor_handler import handle_current_sensors
from .main.helper import config_helper


def create_app(
        engine_id,
        grid_node_address,
        grid_gateway_address,
        data_dir,
        dataset_id,
        cycle_length,
        debug=False,
        test_config=None
):
    """ Create / Configure flask socket application instance.

        Args:
            engine_id : The id of this engine.
            grid_node_address : The address of the local grid node to connect.
            grid_gateway_address : The address of the grid gateway.
            data_dir : The directory containing the engine data.
            dataset_id : The id of the worker dataset to use.
            cycle_length : The length of one engine cycle.
            debug (bool) : debug flag.
            test_config (bool) : Mock database environment.
        Returns:
            app : Flask application instance.
    """
    app = Flask(__name__)
    app.debug = debug
    app.config["SECRET_KEY"] = "justasecretkeythatishouldputhere"

    # Register app blueprints
    app.register_blueprint(html, url_prefix=r"/")

    # Set SQLAlchemy configs
    app = set_database_config(app, test_config=test_config)
    app.app_context().push()
    db.drop_all()
    db.create_all()

    config_helper.engine_id = engine_id
    config_helper.grid_node_address = grid_node_address
    config_helper.grid_gateway_address = grid_gateway_address
    config_helper.data_dir = data_dir
    config_helper.dataset_id = dataset_id

    # start a scheduler regularly reading and saving the sensor data of the workers engines
    start_sensor_scheduler(app, cycle_length=cycle_length)

    return app


def start_sensor_scheduler(app, cycle_length):
    """ Start a scheduler to regularly read in and save the current sensor data.

    :param app: The app context
    :param cycle_length: The length of one cycle in seconds
    :return: None
    """
    logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(handle_current_sensors, 'interval', args=[app, scheduler], seconds=cycle_length)
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
