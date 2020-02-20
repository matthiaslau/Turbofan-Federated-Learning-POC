import json
import os

from flask import render_template
from flask import Response
from flask import send_from_directory

from flask_cors import cross_origin

from . import html
from .helper import config_helper
from .handler.state_handler import get_state
from .handler import sensor_handler
from .handler.stats_handler import get_all_stats
from .persistence import model_manager as mm


# ======= WEB ROUTES ======


@html.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(html.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@html.route("/", methods=["GET"])
def index():
    """Index page."""
    return render_template("index.html")


# ======= WEB ROUTES END ======

# ======= REST API =======


@html.route("/info", methods=["GET"])
@cross_origin()
def info():
    """ Return basic information on this worker like e.g. the ID

       Returns:
            Response : Basic information.
    """
    response = {
        "id": config_helper.engine_id,
        "grid_node_address": config_helper.grid_node_address,
        "grid_gateway_address": config_helper.grid_gateway_address
    }

    return Response(
        json.dumps(response), status=200, mimetype="application/json"
    )


@html.route("/cycle-info", methods=["GET"])
@cross_origin()
def cycle_info():
    """ Return information on the current cycle

       Returns:
            Response : Cycle information.
    """
    response = {
        "state": get_state().name,
        "cycle": sensor_handler.current_cycle,
        "predicted_rul": sensor_handler.current_prediction
    }

    return Response(
        json.dumps(response), status=200, mimetype="application/json"
    )


@html.route("/stats", methods=["GET"])
@cross_origin()
def stats():
    """ Return statistics for the engine

       Returns:
            Response : Statistics information.
    """
    response = get_all_stats()

    return Response(
        json.dumps(response), status=200, mimetype="application/json"
    )


@html.route("/sensor-data", methods=["GET"])
@cross_origin()
def get_sensor_data():
    """ Return all current sensor data.

        Returns:
            Response : List of sensor data from the running series.
    """
    response = mm.list_sensor_data()

    return Response(
        json.dumps(response), status=200, mimetype="application/json"
    )


# ======= REST API END =======
