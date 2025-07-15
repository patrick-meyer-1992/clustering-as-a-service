#!/bin/bash

docker compose -f docker-compose-ci.yml down   && \
docker compose -f docker-compose-ci.yml up   -d --build

sleep 2

pytest integration_tests/