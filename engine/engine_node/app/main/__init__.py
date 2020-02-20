from flask import Blueprint

import syft as sy
import torch as th

hook = sy.TorchHook(th)

html = Blueprint(r"html", __name__)

from . import routes
from .persistence.models import db
