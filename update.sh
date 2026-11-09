#!/bin/bash
set -euo pipefail

DEMO_NAME="ARIES Waybill Demo"
DEMO_DIR_NAME="aries-waybill-demo"

if [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  USER_HOME="$HOME"
fi

if [ "$EUID" -eq 0 ] && [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  RUN_AS_USER="$SUDO_USER"
else
  RUN_AS_USER="$USER"
fi

run_as_user() {
  if [ "$EUID" -eq 0 ] && [ "${SUDO_USER-}" ] && [ "$RUN_AS_USER" != "root" ]; then
    sudo -H -u "$RUN_AS_USER" "$@"
  else
    "$@"
  fi
}

APP_DIR="$USER_HOME/$DEMO_DIR_NAME"

cd "$APP_DIR"

printf '\033[1;32m=========== %s ============\033[0m\n' "$DEMO_NAME"

echo "Installing base build dependencies..."
sudo apt update
sudo apt install -y \
  ca-certificates \
  curl \
  build-essential \
  cmake \
  pkg-config \
  libopencv-dev \
  libomp-dev

echo "Configuring Mobilint APT repository..."
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc
sudo chmod a+r /etc/apt/keyrings/mblt.asc

printf "%s\n" \
  "deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt stable multiverse" | \
  sudo tee /etc/apt/sources.list.d/mobilint.list >/dev/null

sudo apt update

echo "Installing Mobilint runtime packages..."
sudo apt install -y mobilint-qb-runtime mobilint-cli

if apt-cache show mobilint-aries-driver >/dev/null 2>&1; then
  sudo apt install -y mobilint-aries-driver
fi

CREDENTIAL_TIMEOUT="${GIT_CREDENTIAL_TIMEOUT:-3600}"
echo "Configuring Git credential cache (timeout: ${CREDENTIAL_TIMEOUT}s)..."
git config --global credential.helper "cache --timeout=${CREDENTIAL_TIMEOUT}"
if [ "$RUN_AS_USER" != "$USER" ]; then
  run_as_user git config --global credential.helper "cache --timeout=${CREDENTIAL_TIMEOUT}"
fi

echo "Updating repository..."
run_as_user git pull

echo "Starting build..."
run_as_user cmake -S . -B build
run_as_user cmake --build build -j"$(nproc)"

echo "Updating desktop shortcut..."
sudo rm -f /usr/share/applications/aries_waybill_demo.desktop
sudo rm -f /usr/share/applications/aries-waybill-demo.desktop
sudo mkdir -p "$USER_HOME/.local/share/applications/"
sudo cp aries-waybill-demo.desktop "$USER_HOME/.local/share/applications/"
sudo chown "$RUN_AS_USER":"$RUN_AS_USER" "$USER_HOME/.local/share/applications/aries-waybill-demo.desktop" || true

echo "Updating desktop icon..."
sudo mkdir -p "$USER_HOME/.icons/"
sudo cp aries-waybill-demo.png "$USER_HOME/.icons/"
sudo chown "$RUN_AS_USER":"$RUN_AS_USER" "$USER_HOME/.icons/aries-waybill-demo.png" || true

echo "Update completed successfully."