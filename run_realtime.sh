#!/bin/bash
cd "$(dirname "$0")"
source ../.venv/bin/activate
python app_realtime_fusion.py "$@"
