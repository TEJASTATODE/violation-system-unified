from datetime import datetime, timezone

from sensors import rtc


def test_read_rtc_utc(tmp_path, monkeypatch) -> None:
    rtc_dir = tmp_path / "rtc0"
    rtc_dir.mkdir()
    (rtc_dir / "since_epoch").write_text("1791275412\n")
    monkeypatch.setattr(rtc, "RTC_SYSFS", rtc_dir)

    assert rtc.rtc_available() is True
    assert rtc.read_rtc_utc() == datetime.fromtimestamp(1791275412, tz=timezone.utc)


def test_rtc_drift(tmp_path, monkeypatch) -> None:
    rtc_dir = tmp_path / "rtc0"
    rtc_dir.mkdir()
    (rtc_dir / "since_epoch").write_text("0\n")
    monkeypatch.setattr(rtc, "RTC_SYSFS", rtc_dir)
    monkeypatch.setattr(rtc, "datetime", _FixedDateTime)

    assert rtc.rtc_drift_seconds() == 10.0


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(10, tz=tz)
