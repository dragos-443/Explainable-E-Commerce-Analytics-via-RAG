#!/usr/bin/env bash
set -euo pipefail

sudo python3 -m pip install \
  --disable-pip-version-check \
  PyYAML==6.0.2 \
  matplotlib==3.7.5 \
  python-dateutil==2.9.0.post0
