#!/bin/env python
"""
    Turbofan Engine Node emulating a Turbofan Engine
"""
import os

import argparse

from app import create_app

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler


parser = argparse.ArgumentParser(description="Run turbofan engine node application.")

parser.add_argument(
    "--id",
    type=str,
    help="Engine node ID, e.g. --id=alice. Default is os.environ.get('ENGINE_ID', None).",
    default=os.environ.get("ENGINE_ID", None),
)

parser.add_argument(
    "--port",
    "-p",
    type=int,
    help="Port number of the engine server, e.g. --port=8777. Default is os.environ.get('ENGINE_PORT', None).",
    default=os.environ.get("ENGINE_PORT", None),
)

parser.add_argument(
    "--grid_node_address",
    type=str,
    help="Address of the local grid node, e.g. --grid_node_address=0.0.0.0:3000. Default is os.environ.get("
         "'GRID_NODE_ADDRESS','http://0.0.0.0:3000').",
    default=os.environ.get("GRID_NODE_ADDRESS", "0.0.0.0:3000"),
)

parser.add_argument(
    "--grid_gateway_address",
    type=str,
    help="Address used to contact the Grid Gateway. Default is os.environ.get('GRID_GATEWAY_ADDRESS', None).",
    default=os.environ.get("GRID_GATEWAY_ADDRESS", None),
)

parser.add_argument(
    "--data_dir",
    type=str,
    help="The directory of the data to use.",
    default=os.environ.get("DATA_DIR", None),
)

parser.add_argument(
    "--dataset_id",
    type=int,
    help="The ID of the worker dataset splitted from the full training set to use.",
    default=os.environ.get("DATASET_ID", None),
)

parser.add_argument(
    "--cycle_length",
    type=int,
    help="The length of one cycle in seconds.",
    default=os.environ.get("CYCLE_LENGTH", 5),
)

parser.set_defaults(use_test_config=False)


if __name__ == "__main__":
    args = parser.parse_args()

    db_path = "sqlite:///database{}.db".format(args.id)
    app = create_app(
        args.id,
        grid_node_address=args.grid_node_address,
        grid_gateway_address=args.grid_gateway_address,
        data_dir=args.data_dir,
        dataset_id=args.dataset_id,
        cycle_length=args.cycle_length,
        debug=False,
        test_config={"SQLALCHEMY_DATABASE_URI": db_path}
    )

    server = pywsgi.WSGIServer(("", args.port), app, handler_class=WebSocketHandler)
    server.serve_forever()
else:
    # DEPLOYMENT MODE (we use gunicorn's worker to perform load balancing)

    # These environment variables must be set before starting the application.
    engine_id = os.environ.get("ENGINE_ID", None)
    port = os.environ.get("ENGINE_PORT", None)
    grid_node_address = os.environ.get("GRID_NODE_ADDRESS", None)
    grid_gateway_address = os.environ.get("GRID_GATEWAY_ADDRESS", None)
    data_dir = os.environ.get("DATA_DIR", None)
    dataset_id = int(os.environ.get("DATASET_ID", None))
    cycle_length = int(os.environ.get("CYCLE_LENGTH", None))

    app = create_app(
        engine_id,
        grid_node_address=grid_node_address,
        grid_gateway_address=grid_gateway_address,
        data_dir=data_dir,
        dataset_id=dataset_id,
        cycle_length=cycle_length,
        debug=False
    )
