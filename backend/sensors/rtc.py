"""Helpers for the Raspberry Pi 5 built-in RTC."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path


RTC_SYSFS = Path("/sys/class/rtc/rtc0")


def rtc_available() -> bool:
    """Return whether the RTC exposes a readable sysfs epoch."""
    return (RTC_SYSFS / "since_epoch").exists()


def read_rtc_utc() -> datetime:
    """Read the built-in RTC via sysfs without requiring root."""
    seconds = int((RTC_SYSFS / "since_epoch").read_text().strip())
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def rtc_drift_seconds() -> float:
    """Return system UTC minus RTC UTC in seconds."""
    return (datetime.now(timezone.utc) - read_rtc_utc()).total_seconds()


def write_rtc_from_system() -> None:
    """Copy system time into the RTC; the caller must have root privileges."""
    subprocess.run(["hwclock", "--systohc"], check=True)


def ntp_synchronized() -> bool:
    """Return whether systemd reports that NTP is synchronized."""
    try:
        result = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.stdout.strip() == "yes"
