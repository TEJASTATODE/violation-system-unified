"""Background NMEA reader for a u-blox GPS receiver."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

try:
    import pynmea2
    import serial
except ImportError:  # Allows the pipeline to run on development PCs.
    pynmea2 = None
    serial = None


log = logging.getLogger(__name__)


@dataclass
class GPSFix:
    """The latest GPS position and navigation data."""

    valid: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None
    speed_kmh: Optional[float] = None
    course_deg: Optional[float] = None
    utc: Optional[str] = None
    satellites: int = 0
    hdop: Optional[float] = None
    age_s: Optional[float] = None

    def to_dict(self) -> dict:
        """Return a JSON-compatible snapshot."""
        return asdict(self)


class GPSReader:
    """Read NMEA sentences on a daemon thread without blocking callers."""

    def __init__(self, port: str = "/dev/serial0", baud: int = 9600,
                 max_age_s: float = 3.0):
        self.port = port
        self.baud = baud
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._fix = GPSFix()
        self._last_valid_mono = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "GPSReader":
        """Start reading, or remain disabled when dependencies are absent."""
        if serial is None or pynmea2 is None:
            log.warning("pyserial/pynmea2 not installed; GPS disabled")
            return self
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._run, name="gps", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Request that the reader thread stop."""
        self._running = False

    def _run(self) -> None:
        while self._running:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    log.info("GPS opened %s @ %d", self.port, self.baud)
                    while self._running:
                        raw = ser.readline().decode("ascii", errors="ignore").strip()
                        if raw.startswith("$"):
                            self._handle(raw)
            except (serial.SerialException, OSError) as exc:
                log.warning("GPS port error: %s (retry in 2s)", exc)
                time.sleep(2)

    def _handle(self, line: str) -> None:
        try:
            msg = pynmea2.parse(line)
        except pynmea2.ParseError:
            return

        if isinstance(msg, pynmea2.types.talker.RMC):
            with self._lock:
                if msg.status == "A":
                    self._fix.valid = True
                    self._fix.lat = msg.latitude
                    self._fix.lon = msg.longitude
                    self._fix.speed_kmh = (msg.spd_over_grnd or 0.0) * 1.852
                    self._fix.course_deg = msg.true_course
                    if msg.datestamp and msg.timestamp:
                        self._fix.utc = datetime.combine(
                            msg.datestamp, msg.timestamp, tzinfo=timezone.utc
                        ).isoformat()
                    self._last_valid_mono = time.monotonic()
                else:
                    self._fix.valid = False
        elif isinstance(msg, pynmea2.types.talker.GGA):
            with self._lock:
                self._fix.satellites = int(msg.num_sats or 0)
                self._fix.hdop = float(msg.horizontal_dil) if msg.horizontal_dil else None

    def get(self) -> GPSFix:
        """Return a thread-safe snapshot, invalidating stale fixes."""
        with self._lock:
            fix = GPSFix(**asdict(self._fix))
            last_valid_mono = self._last_valid_mono
        if last_valid_mono:
            fix.age_s = time.monotonic() - last_valid_mono
            if fix.age_s > self.max_age_s:
                fix.valid = False
        return fix
