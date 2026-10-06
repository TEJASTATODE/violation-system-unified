"""Environment-backed settings for the traffic violation pipeline."""

import os


GPS_ENABLED = os.getenv("GPS_ENABLED", "1") == "1"
GPS_PORT = os.getenv("GPS_PORT", "/dev/serial0")
GPS_BAUD = int(os.getenv("GPS_BAUD", "9600"))
GPS_MAX_AGE_S = float(os.getenv("GPS_MAX_AGE_S", "3.0"))
CLOCK_MAX_DRIFT_S = float(os.getenv("CLOCK_MAX_DRIFT_S", "2.0"))
GPS_SYNC_TIMEOUT_S = int(os.getenv("GPS_SYNC_TIMEOUT_S", "300"))
