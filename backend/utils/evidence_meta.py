"""Write metadata next to captured evidence images."""

import json
from datetime import datetime, timezone
from pathlib import Path


def write_meta(evidence_jpg: str, violation_type: str, fix,
               clock_source: str = "unknown") -> None:
    """Write a JSON sidecar containing the capture clock and GPS snapshot."""
    meta = {
        "violation": violation_type,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "clock_source": clock_source,
        "gps_valid": fix.valid,
        "lat": fix.lat,
        "lon": fix.lon,
        "speed_kmh": fix.speed_kmh,
        "course_deg": fix.course_deg,
        "satellites": fix.satellites,
        "hdop": fix.hdop,
        "gps_utc": fix.utc,
    }
    Path(evidence_jpg).with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
