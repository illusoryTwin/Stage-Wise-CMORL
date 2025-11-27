#!/bin/bash
xhost +local:docker
docker compose run --rm stage-wise-cmorl
