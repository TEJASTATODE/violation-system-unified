"""Synchronize the system clock and Pi RTC from the best available source."""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from config import CLOCK_MAX_DRIFT_S, GPS_BAUD, GPS_PORT, GPS_SYNC_TIMEOUT_S
from sensors.gps import GPSReader
from sensors.rtc import (
    ntp_synchronized,
    read_rtc_utc,
    rtc_available,
    write_rtc_from_system,
)


log = logging.getLogger(__name__)
CLOCK_SOURCE_PATH = Path("/run/clock_source")


def _write_clock_source(source: str) -> None:
    try:
        CLOCK_SOURCE_PATH.write_text(source + "\n")
    except OSError as exc:
        log.warning("Could not write %s: %s", CLOCK_SOURCE_PATH, exc)


def _set_system_time(utc: datetime) -> None:
    value = utc.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    subprocess.run(["date", "-u", "-s", value], check=True)


def main() -> int:
    """Run the non-fatal boot-time synchronization procedure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not rtc_available():
        log.info("RTC unavailable; leaving system clock unchanged (source=rtc)")
        _write_clock_source("rtc")
        return 0

    if ntp_synchronized():
        write_rtc_from_system()
        log.info("Clock source: ntp")
        _write_clock_source("ntp")
        return 0

    gps = GPSReader(GPS_PORT, GPS_BAUD).start()
    deadline = time.monotonic() + GPS_SYNC_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            fix = gps.get()
            if fix.valid and fix.utc:
                gps_time = datetime.fromisoformat(fix.utc)
                drift = abs((gps_time - datetime.now(timezone.utc)).total_seconds())
                if drift > CLOCK_MAX_DRIFT_S:
                    _set_system_time(gps_time)
                write_rtc_from_system()
                log.info("Clock source: gps (drift before sync %.3fs)", drift)
                _write_clock_source("gps")
                return 0
            time.sleep(1)
    finally:
        gps.stop()

    log.warning("GPS did not acquire a fix; keeping RTC-provided boot time (source=rtc)")
    _write_clock_source("rtc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
