#!/bin/sh

set -e

#docker build -t syft-base ./docker_syft_base
docker build -t turbofan-engine ./engine
docker build -t turbofan-federated-trainer ./federated_trainer
docker build -t pysyft-notebook ./pysyft-notebook
