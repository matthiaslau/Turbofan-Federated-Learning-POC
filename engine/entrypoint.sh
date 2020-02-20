#!/bin/bash
exec gunicorn -b=:${PORT} --config /app/gunicorn.conf -k flask_sockets.worker --chdir /app/grid_node websocket_app:app &
exec gunicorn -b=:${ENGINE_PORT} --config /app/gunicorn.conf --log-config /app/logging.conf -k flask_sockets.worker --chdir /app/engine_node turbofan_worker:app
