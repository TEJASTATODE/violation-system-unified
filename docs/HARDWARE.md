# Raspberry Pi 5 Hardware

## Wiring

| NEO-6M wire | Raspberry Pi 5 | Header pin |
| --- | --- | --- |
| TX | GPIO15 / RXD | 10 |
| RX | GPIO14 / TXD | 8 |
| VCC | 3.3V | 1 |
| GND | Ground | 6 |

The Hailo AI HAT uses PCIe and does not use UART or I2C. Its board covers the
40-pin header, so use a stacking header or solder wires to pins 1, 6, 8, and 10.

The Pi 5 built-in RTC is used through `/sys/class/rtc/rtc0`; this project does
not add DS3231 or other I2C RTC code.

## Setup

From the repository directory on the Pi, run `sudo deploy/setup_pi.sh`. Confirm
the rechargeable-battery prompt only when the battery connected to J5 is a
rechargeable type. The script enables UART, installs dependencies, creates the
virtual environment, and enables the clock-sync and pipeline services.

Useful checks:

```sh
cat /dev/serial0
sudo hwclock -r
timedatectl
```

The pipeline runs as user `pi` and writes evidence JSON beside each evidence
JPEG. `python -m sensors.time_sync` must run as root when it needs to set the
system clock or write the RTC.

## Troubleshooting

- No NMEA data: serial is not enabled, or GPS TX/RX are not crossed.
- No fix: place the antenna outdoors; a cold start can take up to five minutes.
- Time resets after power-off: the RTC battery is missing or dead.
- Permission denied on `/dev/serial0`: add the user to `dialout`, then log in again.
- On a development PC without GPS packages or `/dev/serial0`, GPS remains
  disabled and evidence records `gps_valid: false`.
