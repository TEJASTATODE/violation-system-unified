"""Print live GPS snapshots for a short hardware check."""

import logging
import time

from config import GPS_BAUD, GPS_MAX_AGE_S, GPS_PORT
from sensors.gps import GPSReader


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    reader = GPSReader(GPS_PORT, GPS_BAUD, GPS_MAX_AGE_S).start()
    try:
        for _ in range(30):
            logging.info("GPS fix: %s", reader.get().to_dict())
            time.sleep(1)
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
