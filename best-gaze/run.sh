#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

VENV="$DIR/venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
    echo "Upgrading pip..."
    "$VENV/bin/pip" install --upgrade pip
fi

echo "Installing dependencies..."
"$VENV/bin/pip" install --no-compile -q \
    opencv-python mediapipe pyautogui SpeechRecognition sounddevice numpy \
    "pyobjc-core==11.1" \
    "pyobjc-framework-Quartz==11.1" \
    "pyobjc-framework-Cocoa==11.1" \
    "pyobjc-framework-Vision==11.1"

echo "Launching eye tracker..."
exec "$VENV/bin/python" main.py
