#!/bin/bash

docker run --name test-mongo -p 27018:27017 -d mongo
pytest
docker stop test-mongo && docker rm test-mongo