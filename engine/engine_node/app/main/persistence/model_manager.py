from sqlalchemy.exc import SQLAlchemyError

from .models import db, SensorData


def _get_all_sensor_data_in_db():
    """ Retrieve all sensor data objects from the database.

    :return: All sensor data objects
    """
    return db.session.query(SensorData).all()


def _remove_all_sensor_data_in_db():
    """ Remove all sensor data from the database. """
    SensorData.query.delete()
    db.session.commit()


def delete_sensor_data():
    """ Remove all sensor data.

    :return:
        A dict with structure: {"success": Bool}.
        On error returns dict: {"success": Bool, "error": error message}.
    """

    try:
        _remove_all_sensor_data_in_db()
        return {"success": True}
    except SQLAlchemyError:
        return {
            "success": False,
            "error": "Something went wrong while deleting the sensor data.",
        }


def list_sensor_data():
    """ List all sensor data.

    :return:
        A dict with structure: {"success": Bool, "sensor_data": List}.
        On error returns dict: {"success": Bool, "error": error message}.
    """
    try:
        result = _get_all_sensor_data_in_db()
        sensor_data = [data.as_dict() for data in result]
        return {"success": True, "sensor_data": sensor_data}
    except SQLAlchemyError as e:
        return {"success": False, "error": str(e)}
