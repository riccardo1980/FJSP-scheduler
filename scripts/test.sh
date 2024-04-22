#!/usr/bin/env bash
set -e

SRCS="demo_scheduler"

[ -d $SRCS ] || (echo "Run this script from project root"; exit 1)

set -x

poetry run coverage run -m pytest
poetry run coverage report --show-missing