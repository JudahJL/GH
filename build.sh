#!/usr/bin/env bash
set -o errexit

uv sync

python manage.py collectstatic --no-input

python manage.py migrate