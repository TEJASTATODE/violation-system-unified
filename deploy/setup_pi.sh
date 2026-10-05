#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/pi/violation-system-unified"
CONFIG_FILE="/boot/firmware/config.txt"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run this script with sudo."
    exit 1
fi

raspi-config nonint do_serial_hw 0
raspi-config nonint do_serial_cons 1
usermod -aG dialout pi

read -r -p "Is the RTC battery rechargeable? Add dtparam=rtc_bbat_vchg=3000000? [y/N] " answer
if [[ "$answer" =~ ^[Yy]$ ]] && ! grep -q '^dtparam=rtc_bbat_vchg=3000000$' "$CONFIG_FILE"; then
    printf '\ndtparam=rtc_bbat_vchg=3000000\n' >> "$CONFIG_FILE"
fi

python3 -m venv "$PROJECT_DIR/venv"
"$PROJECT_DIR/venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
install -m 0644 "$PROJECT_DIR/deploy/gps-timesync.service" /etc/systemd/system/
install -m 0644 "$PROJECT_DIR/deploy/violation-pipeline.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable gps-timesync.service violation-pipeline.service

echo "Setup complete. Reboot the Pi to apply UART and RTC configuration."
