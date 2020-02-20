#!/bin/bash
exec gunicorn --config /gunicorn.conf --log-config /logging.conf -k flask_sockets.worker websocket_app:app \
"$@"
