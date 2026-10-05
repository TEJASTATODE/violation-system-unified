"""Report Raspberry Pi RTC and system clock status."""

import logging
from datetime import datetime, timezone

from sensors.rtc import ntp_synchronized, read_rtc_utc, rtc_available, rtc_drift_seconds


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not rtc_available():
        logging.info("RTC unavailable")
        return
    logging.info("RTC UTC: %s", read_rtc_utc().isoformat())
    logging.info("System UTC: %s", datetime.now(timezone.utc).isoformat())
    logging.info("Drift seconds: %.3f", rtc_drift_seconds())
    logging.info("NTP synchronized: %s", ntp_synchronized())


if __name__ == "__main__":
    main()
