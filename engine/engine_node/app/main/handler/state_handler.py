from enum import Enum


class State(Enum):
    STOPPED = 0
    RUNNING = 1
    MAINTENANCE = 2
    FAILURE = 3


_current_state = State.STOPPED


def get_state():
    """ Return the current state of the engine.

    :return: The current state
    """
    return _current_state


def set_state(state: State):
    """ Set the current state of the engine.

    :param state: The new engine state
    """
    global _current_state
    _current_state = state
