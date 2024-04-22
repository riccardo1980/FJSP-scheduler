#!/usr/bin/env bash
set -e

SRCS="demo_scheduler"
TEST_SRCS="tests"

[ -d "$SRCS" ] || (echo "Run this script from project root"; exit 1)

set -x

poetry run black "$SRCS" "$TEST_SRCS"
poetry run isort "$SRCS" "$TEST_SRCS"
poetry run mypy "$SRCS" "$TEST_SRCS"
poetry run flake8 "$SRCS" "$TEST_SRCS"