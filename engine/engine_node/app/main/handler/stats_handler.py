from enum import Enum


class Stats(Enum):
    FAILURES = 0
    PREVENTED_FAILURES = 1
    PREVENTED_FAILURES_TOO_EARLY = 2


stats = {
    Stats.FAILURES.name: 0,
    Stats.PREVENTED_FAILURES.name: 0,
    Stats.PREVENTED_FAILURES_TOO_EARLY.name: 0,
}


def track(stat_type: Stats):
    """ Track a new event from the given stats type.

    :param stat_type: The stats type to track
    """
    stats[stat_type.name] += 1


def get_all_stats():
    """ Retrieve all stats for the engine.

    :return: A dictionary with all stats
    """
    return stats


def get_stats(stat_type: Stats):
    """ Retrieve the event count of a given stat.

    :param stat_type: The stat to retrieve
    :return: The requested stats
    """
    return stats[stat_type.name]
