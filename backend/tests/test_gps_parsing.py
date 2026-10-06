import pytest

from sensors.gps import GPSReader


def nmea(body: str) -> str:
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    return f"${body}*{checksum:02X}"


def test_rmc_and_gga_parse_with_conversion() -> None:
    reader = GPSReader()
    reader._handle(nmea("GNRMC,083012.00,A,4907.038,N,01131.000,E,10.0,84.4,061026,,,A"))
    reader._handle(nmea("GNGGA,083012.00,4907.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"))

    fix = reader.get()
    assert fix.valid is True
    assert fix.lat == pytest.approx(49.1173)
    assert fix.lon == pytest.approx(11.5166667)
    assert fix.speed_kmh == pytest.approx(18.52)
    assert fix.course_deg == pytest.approx(84.4)
    assert fix.satellites == 8
    assert fix.hdop == pytest.approx(0.9)
    assert fix.utc == "2026-10-06T08:30:12+00:00"


def test_rmc_status_v_invalidates_fix() -> None:
    reader = GPSReader()
    reader._handle(nmea("GPRMC,083012.00,A,4907.038,S,01131.000,W,0.0,0.0,061026,,,A"))
    assert reader.get().lat == pytest.approx(-49.1173)
    assert reader.get().lon == pytest.approx(-11.5166667)
    reader._handle(nmea("GPRMC,083013.00,V,,,,,,,061026,,,N"))
    assert reader.get().valid is False


def test_stale_fix_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = GPSReader(max_age_s=3.0)
    reader._handle(nmea("GPRMC,083012.00,A,4907.038,N,01131.000,E,1.0,0.0,061026,,,A"))
    now = reader._last_valid_mono
    monkeypatch.setattr("sensors.gps.time.monotonic", lambda: now + 4.0)
    fix = reader.get()
    assert fix.valid is False
    assert fix.age_s == pytest.approx(4.0)
