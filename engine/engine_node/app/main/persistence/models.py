from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class SensorData(db.Model):
    """Database table that stores current sensor data."""

    __tablename__ = "sensor_data"

    engine_no = db.Column(db.Integer, primary_key=True)
    time_in_cycles = db.Column(db.Integer, primary_key=True)

    operational_setting_1 = db.Column(db.Float)
    operational_setting_2 = db.Column(db.Float)
    operational_setting_3 = db.Column(db.Float)

    sensor_measurement_1 = db.Column(db.Float)
    sensor_measurement_2 = db.Column(db.Float)
    sensor_measurement_3 = db.Column(db.Float)
    sensor_measurement_4 = db.Column(db.Float)
    sensor_measurement_5 = db.Column(db.Float)
    sensor_measurement_6 = db.Column(db.Float)
    sensor_measurement_7 = db.Column(db.Float)
    sensor_measurement_8 = db.Column(db.Float)
    sensor_measurement_9 = db.Column(db.Float)
    sensor_measurement_10 = db.Column(db.Float)
    sensor_measurement_11 = db.Column(db.Float)
    sensor_measurement_12 = db.Column(db.Float)
    sensor_measurement_13 = db.Column(db.Float)
    sensor_measurement_14 = db.Column(db.Float)
    sensor_measurement_15 = db.Column(db.Float)
    sensor_measurement_16 = db.Column(db.Float)
    sensor_measurement_17 = db.Column(db.Float)
    sensor_measurement_18 = db.Column(db.Float)
    sensor_measurement_19 = db.Column(db.Float)
    sensor_measurement_20 = db.Column(db.Float)
    sensor_measurement_21 = db.Column(db.Float)
    sensor_measurement_22 = db.Column(db.Float)
    sensor_measurement_23 = db.Column(db.Float)

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
